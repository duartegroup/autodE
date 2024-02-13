"""
The theory behind this original NEB implementation is taken from
Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)
"""
import numpy as np

from typing import Optional, Sequence, List, Any, TYPE_CHECKING, Union, Type
from copy import deepcopy

from autode.log import logger
from autode.calculations import Calculation
from autode.wrappers.methods import Method
from autode.species.species import Species
from autode.input_output import xyz_file_to_molecules
from autode.path import Path
from autode.utils import work_in, ProcessPool
from autode.config import Config
from autode.neb.idpp import IDPP
from scipy.optimize import minimize
from autode.values import Distance, PotentialEnergy, ForceConstant

if TYPE_CHECKING:
    from autode.wrappers.methods import Method
    from autode.neb.ci import CImages


def energy_gradient(image, method, n_cores):
    """Calculate energies and gradients for an image using a EST method"""

    if isinstance(method, Method):
        return _est_energy_gradient(image, method, n_cores)

    elif isinstance(method, IDPP):
        return _idpp_energy_gradient(image, method, n_cores)

    raise ValueError(
        f"Cannot calculate energy and gradient with {method}."
        "Must be one of: ElectronicStructureMethod, IDPP"
    )


def _est_energy_gradient(image, est_method, n_cores):
    """Electronic structure energy and gradint"""
    calc = Calculation(
        name=f"{image.name}_{image.iteration}",
        molecule=image,
        method=est_method,
        keywords=est_method.keywords.grad,
        n_cores=n_cores,
    )

    @work_in(image.name)
    def run():
        calc.run()

    run()
    return image


def _idpp_energy_gradient(
    image: "Image",
    idpp: IDPP,
    n_cores: int,
) -> "Image":
    """
    Evaluate the energy and gradient of an image using an image dependent
    pair potential IDDP instance and set the energy and gradient on the image

    ---------------------------------------------------------------------------
    Arguments:
        image: Image in the NEB

        idpp: Instance

        n_cores: *UNUSED*

    Returns:
        (autode.neb.original.Image): Image
    """
    image.energy = idpp(image)
    image.gradient = idpp.grad(image)

    return image


def total_energy(flat_coords, images, method, n_cores, plot_energies):
    """Compute the total energy across all images"""
    images.set_coords(flat_coords)

    # Number of cores per process is the floored total divided by n images
    # minus the two end points that will be fixed
    n_cores_pp = 1
    if len(images) > 2:
        n_cores_pp = max(int(n_cores // (len(images) - 2)), 1)

    logger.info(
        f"Calculating energy and forces for all images with "
        f"{n_cores} total cores and {n_cores_pp} per process"
    )

    # Run an energy + gradient evaluation across all images (parallel for EST)
    if isinstance(method, IDPP):
        images[1:-1] = [
            energy_gradient(images[i], method, n_cores_pp)
            for i in range(1, len(images) - 1)
        ]
    else:
        with ProcessPool(max_workers=n_cores) as pool:
            results = [
                pool.submit(energy_gradient, images[i], method, n_cores_pp)
                for i in range(1, len(images) - 1)
            ]

            images[1:-1] = [res.result() for res in results]

    images.increment()

    if plot_energies:
        images.plot_energies()

    all_energies = [image.energy for image in images]
    rel_energies = [energy - min(all_energies) for energy in all_energies]

    logger.info(f"Path energy = {sum(rel_energies):.5f}")
    return sum(rel_energies)


def derivative(flat_coords, images, method, n_cores, plot_energies):
    """
    Compute the derivative of the total energy with respect to all
    components. Several arguments are unused as SciPy requires the jacobian
    function to have the same signature as the function that's being minimised.
    See: https://tinyurl.com/scipyopt
    """

    # Forces for the first image are fixed at zero
    forces = np.zeros(shape=images[0].gradient.shape)

    # No need to calculate gradient as should already be there from energy eval
    for i in range(1, len(images) - 1):
        force = images[i].get_force(im_l=images[i - 1], im_r=images[i + 1])
        forces = np.append(forces, force)

    # Final zero set of forces
    forces = np.append(forces, np.zeros(shape=images[-1].gradient.shape))

    # dV/dx is negative of the force
    logger.info(f"|F| = {np.linalg.norm(forces):.4f} Ha Å-1")
    return -forces


class Image(Species):
    def __init__(
        self,
        species: Species,
        name: str,
        k: ForceConstant,
    ):
        """
        Image in a NEB

        --------------------------------------------------------------------------------
        Arguments:
            species (Species): Molecule for which this image represents

            name (str): Name of this image

            k (ForceConstant): Force constant of the harmonic potential joining this
                               image to its neighbour(s)
        """
        super().__init__(
            name=name,
            charge=species.charge,
            mult=species.mult,
            atoms=species.atoms.copy(),
        )
        self.solvent = deepcopy(species.solvent)
        self.energy = deepcopy(species.energy)

        self.iteration = 0  #: Current optimisation iteration of this image
        self.k = k

    def _generate_conformers(self, *args, **kwargs):
        raise RuntimeError("Cannot create conformers of an image")

    def _tau_xl_x_xr(
        self,
        im_l: "Image",
        im_r: "Image",
    ) -> tuple:
        """
        Calculate the normalised τ vector, along with the coordinates of the
        left, this and right images

        -----------------------------------------------------------------------
        Arguments:
            im_l: (autode.neb.Image)
            im_r: (autode.neb.Image)

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        assert self.energy is not None, "Energy must be set to calculate tau"
        assert im_l.energy is not None, "Left image energy must be set"
        assert im_r.energy is not None, "Right image energy must be set"

        # ΔV_i^max
        dv_max = max(
            np.abs(im_r.energy - self.energy),
            np.abs(im_l.energy - self.energy),
        )

        # ΔV_i^min
        dv_min = min(
            np.abs(im_r.energy - self.energy),
            np.abs(im_l.energy - self.energy),
        )

        # x_i-1,   x_i,   x_i+1
        x_l, x, x_r = [
            image.coordinates.flatten() for image in (im_l, self, im_r)
        ]
        # τ_i+
        tau_plus = x_r - x
        # τ_i-
        tau_minus = x - x_l

        if im_l.energy < self.energy < im_r.energy:
            tau = tau_plus

        elif im_r.energy < self.energy < im_l.energy:
            tau = tau_minus

        elif im_l.energy < im_r.energy:
            tau = tau_plus * dv_max + tau_minus * dv_min

        elif im_r.energy < im_l.energy:
            tau = tau_plus * dv_min + tau_minus * dv_max

        else:
            raise RuntimeError("Something went very wrong in the NEB!")

        # Normalised τ vector and coordinates of the images
        return tau / np.linalg.norm(tau), x_l, x, x_r

    def get_force(
        self,
        im_l: "Image",
        im_r: "Image",
    ) -> np.ndarray:
        """
        Compute F_i. Notation from:
        Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)

        also a copy in autode/common

        -----------------------------------------------------------------------
        Arguments:
            im_l (autode.neb.Image): Left image (i-1)
            im_r (autode.neb.Image): Right image (i+1)
        """
        assert self.gradient is not None, "Gradient must be set to calc force"

        # τ,  x_i-1,  x_i,   x_i+1
        hat_tau, x_l, x, x_r = self._tau_xl_x_xr(im_l, im_r)

        # F_i^s||
        f_parallel = (
            np.linalg.norm(x_r - x) * im_r.k - np.linalg.norm(x - x_l) * im_l.k
        ) * hat_tau

        # ∇V(x)_i|_|_ = ∇V(x)_i - (∇V(x)_i•τ) τ
        grad_perp = self.gradient - np.dot(self.gradient, hat_tau) * hat_tau

        # F_i = F_i^s|| -  ∇V(x)_i|_|_
        return f_parallel - grad_perp

    @property
    def gradient(self) -> Optional[np.ndarray]:
        return None if self._grad is None else self._grad.flatten()  # type: ignore

    @gradient.setter
    def gradient(self, value: Optional[np.ndarray]):
        self._grad = None if value is None else value.flatten()


class Images(Path):
    def __init__(
        self,
        init_k: ForceConstant,
        min_k: Optional[ForceConstant] = None,
        max_k: Optional[ForceConstant] = None,
    ):
        """
        Set of images joined by harmonic springs with force constant k

        -----------------------------------------------------------------------
        Arguments:

            init_k (ForceConstant): Initial force constant

            min_k (ForceConstant | None): Minimum value of k

            max_k (ForceConstant | None): Maximum value of k
        """
        super().__init__()

        self.init_k = init_k
        self.min_k = init_k / 10 if min_k is None else min_k
        self.max_k = 2 * init_k if max_k is None else max_k
        assert (
            self.max_k > self.min_k
        ), "Can't set the min force constant above the max"

    def __eq__(self, other):
        """Equality od two climbing image NEB paths"""
        if not isinstance(other, Images):
            return None

        if any((self.min_k != other.min_k, self.max_k != other.max_k)):
            return False

        return super().__eq__(other)

    def increment(self):
        """Advance all the iteration numbers on the images to name correctly
        also update force constants"""

        for image in self:
            image.iteration += 1

        if Config.adaptive_neb_k and all(im.energy is not None for im in self):
            logger.info("Updating force constants")
            # Notation from https://doi.org/10.1063/1.1329672
            delta_k = self.max_k - self.min_k

            # E_ref is the maximum energy of the end points
            energies = [image.energy for image in self]
            e_ref = max(energies[0], energies[-1])
            e_max = max(energies)

            if e_ref == e_max:
                logger.warning(
                    "Cannot adjust k, the reference energy was the " "maximum"
                )
                # Return otherwise we'll divide by zero here
                return

            for image in self:
                if image.energy < e_ref:
                    image.k = self.min_k

                else:
                    image.k = self.max_k - delta_k * float(
                        (e_max - image.energy) / (e_max - e_ref)
                    )
        return None

    def plot_energies(
        self, save=False, name="None", color=None, xlabel="NEB coordinate"
    ):
        """Plot the NEB surface"""
        import matplotlib.pyplot as plt

        blues = plt.get_cmap("Blues")

        color = (
            blues((self[0].iteration + 1) / 20)
            if color is None
            else str(color)
        )
        super().plot_energies(save, name, color, xlabel)

    def coords(self):
        """Get a flat array of all components of every atom"""
        coords = np.array([])
        for image in self:
            coords = np.append(coords, image.coordinates.flatten())
        return coords

    def set_coords(self, coords):
        """
        Set the flat array of coordinates to the species in the images

        -----------------------------------------------------------------------
        Arguments:
            coords (np.ndarray): shape (num x n x 3,)
        """

        n_atoms = self[0].n_atoms
        coords = coords.reshape((len(self), n_atoms, 3))

        for i, image in enumerate(self):
            image.coordinates = coords[i]

        return None

    def append_species(self, species: Species) -> None:
        """Add a species to the list of images"""
        super().append(
            Image(species=species, name=f"{len(self)}", k=self.init_k)
        )

    def copy(self) -> "Images":
        return deepcopy(self)


class NEB:
    _images_type: Union[Type[Images], Type["CImages"]] = Images

    def __init__(
        self,
        init_k: ForceConstant = ForceConstant(0.1, units="Ha / Å^2"),
        **kwargs,
    ):
        """
        Nudged elastic band (NEB)

        Warning: The initial/final species or those in a species list must have
        the same atom ordering.

        -----------------------------------------------------------------------
        Arguments:
            init_k: Initial force constant between each image
        """
        self._raise_exception_if_any(kwargs)
        self._init_k = init_k
        self.images = Images(init_k=init_k)

    @property
    def init_k(self) -> ForceConstant:
        """Initial force constant used to in this NEB"""
        return self._init_k

    @classmethod
    def from_file(
        cls,
        filename: str,
        init_k: Optional[float] = None,
    ) -> "NEB":
        """
        Create a nudged elastic band from a .xyz file containing multiple
        images.
        """

        molecules = xyz_file_to_molecules(filename)
        if init_k is None and all(m.energy is not None for m in molecules):
            logger.info(
                "Have a set of energies from file. Can adaptively "
                "choose a sensible force constant (k)"
            )

            max_de = max(
                abs(molecules[i].energy - molecules[i + 1].energy)  # type: ignore
                for i in range(len(molecules) - 1)
            )

            # TODO: test reasonableness of this function...
            # use a shifted tanh to interpolate in [0.005, 0.2005]
            init_k = ForceConstant(
                0.1 * (np.tanh((max_de.to("kcal mol-1") - 40) / 20) + 1)
                + 0.005,
                units="Ha / Å^2",
            )

        if init_k is None:  # choose a sensible default
            init_k = 0.1

        logger.info(
            f"Using k = {init_k:.6f} Ha Å^-1 as the NEB force constant"
        )
        return cls.from_list(molecules, init_k=ForceConstant(init_k))

    @classmethod
    def from_list(
        cls,
        species_list: Sequence[Species],
        init_k: ForceConstant = ForceConstant(0.1, units="Ha / Å^2"),
    ) -> "NEB":
        """
        Nudged elastic band constructed from list of species

        -----------------------------------------------------------------------
        Arguments:
            species_list: Full set of initial images that will form the while NEB

            init_k: Force constant

        Returns:
            (NEB):
        """
        neb = cls(init_k=init_k)

        for species in species_list:
            neb.images.append_species(species)

        logger.info(f"Initialised a NEB with {len(neb.images)} images")
        return neb

    @classmethod
    def from_end_points(
        cls,
        initial: Species,
        final: Species,
        num: int,
        init_k: ForceConstant = ForceConstant(0.1, units="Ha / Å^2"),
    ) -> "NEB":
        """
        Construct a nudged elastic band from only the endpoints. The atomic
        ordering must be identical in the initial and final species

        -----------------------------------------------------------------------
          Arguments:
              initial: Initial/left-most species in the NEB

              final: Final/right-most species in the NEB

              num: Number of images to create

              init_k: Initial force constant

          Returns:
              (NEB):
        """

        if initial.sorted_atomic_symbols != final.sorted_atomic_symbols:
            raise ValueError(
                "Cannot construct a NEB from species with different atoms"
            )

        neb = cls.from_list(
            species_list=cls._interpolated_species(initial, final, n=num),
            init_k=init_k,
        )
        neb.idpp_relax()

        return neb

    def _minimise(self, method, n_cores, etol, max_n=30) -> Any:
        """Minimise the energy of every image in the NEB"""
        logger.info(f"Minimising to ∆E < {etol:.4f} Ha on all NEB coordinates")

        result = minimize(
            total_energy,
            x0=self.images.coords(),
            method="L-BFGS-B",
            jac=derivative,
            args=(self.images, method, n_cores, True),
            tol=etol,
            options={"maxfun": max_n},
        )

        logger.info(f"NEB path energy = {result.fun:.5f} Ha, {result.message}")
        return result

    def partition(
        self,
        max_delta: Distance,
        distance_idxs: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Partition this NEB such that there are no distances between images
        exceeding max_delta. Will run IDPP (image dependent pair potential)
        relaxations on intermediate images.

        -----------------------------------------------------------------------
        Arguments:
            max_delta: The maximum allowed max_atoms(|x_k - x_k+1|) where
                       x_k are the cartesian coordinates of the k-th NEB
                       image and the maximum is over the atom-wise distance

            distance_idxs: Indexes of atoms used to calculate the max_delta.
                           If none then all distances are used. For example if
                           only distance_idxs = [0] then |x_k,0 - x_k+1,0|
                           will be calculated, where 0 is the atom index and
                           k is the image index
        """
        logger.info("Interpolating")

        assert len(self.images) > 1
        _list = []

        for i, left_image in enumerate(self.images[:-1]):
            right_image = self.images[i + 1]

            n = 2
            sub_neb = NEB.from_end_points(left_image, right_image, num=n)

            while (
                sub_neb._max_atom_distance_between_images(distance_idxs)
                > max_delta
            ):
                try:
                    sub_neb = NEB.from_end_points(
                        left_image, right_image, num=n
                    )
                except RuntimeError:
                    logger.warning("Failed to IDPP relax the interpolated NEB")

                n += 1

            for image in sub_neb.images[:-1]:  # add all apart from the last
                _list.append(image)

        _list.append(self.images[-1])  # end with the last
        self.images.clear()

        for image in _list:
            self.images.append_species(image)

        logger.info(
            f"Partition successful – now have {len(self.images)} " f"images"
        )
        return None

    def print_geometries(self, name="neb") -> None:
        return self.images.print_geometries(name)

    @staticmethod
    def _interpolated_species(
        initial: Species, final: Species, n: int
    ) -> List[Species]:
        """Generate simple interpolated coordinates for these set of images
        in Cartesian coordinates"""

        if n < 2:
            raise RuntimeError("Cannot interpolated 2 images to <2")

        if n == 2:
            return [initial.copy(), final.copy()]

        intermediate_species = []

        # Interpolate images between the starting point i=0 and end point i=n-1
        for i in range(1, n - 1):
            # Use a copy of the starting point for atoms, charge etc.
            species: Species = initial.copy()

            # For all the atoms in the species translate an amount so the
            # spacing is even between the initial and final points
            for j, atom in enumerate(species.atoms):
                # Shift vector is final minus current
                shift = final.atoms[j].coord - atom.coord
                # then an equal spacing is the i-th point in the grid
                atom.translate(vec=shift * (i / n))

            intermediate_species.append(species)

        return [initial.copy()] + intermediate_species + [final.copy()]

    @work_in("neb")
    def calculate(
        self,
        method: "Method",
        n_cores: int,
        name_prefix: str = "",
        etol_per_image: Union[float, PotentialEnergy] = PotentialEnergy(
            0.6, units="kcal mol-1"
        ),
    ) -> None:
        """
        Optimise the NEB using forces calculated from electronic structure

        -----------------------------------------------------------------------
        Arguments:
            method: Method used to calculate the energy and gradient. Will
                    use method.keywords.grad keywords

            n_cores: Number of cores to use for the calculation

            name_prefix: Prefix for the naming of the geometry and plot
                         generated by this function

            etol_per_image: Energy tolerance per image to use in the L-BFGS-B
                            minimisation
        """
        import matplotlib.pyplot as plt

        self.print_geometries(name=f"{name_prefix}neb_init")

        # Calculate energy on the first and final points as these will not be recalc-ed
        for idx in [0, -1]:
            energy_gradient(self.images[idx], method=method, n_cores=n_cores)

        if isinstance(etol_per_image, PotentialEnergy):
            etol_per_image = float(
                etol_per_image.to("Ha")
            )  # use float for scipy

        result = self._minimise(
            method, n_cores, etol=etol_per_image * len(self.images)
        )

        # Set the optimised coordinates for all the images
        self.images.set_coords(result.x)
        self.print_geometries(name=f"{name_prefix}neb_optimised")

        # and save the plot
        plt.savefig(f"{name_prefix}neb_optimised.pdf")
        plt.close()
        return None

    @property
    def peak_species(self) -> Optional[Species]:
        """TS guess species for this NEB: highest energy saddle point"""
        if not self.images.contains_peak:
            logger.warning("Found no peaks in the NEB")
            return None

        assert (
            self.images.peak_idx is not None
        ), "Must have a peak index with a peak"
        image = self.images[self.images.peak_idx]

        return image.new_species()

    def idpp_relax(self) -> None:
        """
        Relax the NEB using the image dependent pair potential

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`IDPP <autode.neb.idpp.IDPP.__init__>`
        """
        logger.info(f"Minimising NEB with IDPP potential")

        images = self.images.copy()
        images.min_k = images.max_k = ForceConstant(0.1, units="Ha / Å^2")
        idpp = IDPP(images=images)

        for i, image in enumerate(images):
            image.energy = idpp(image)
            image.gradient = idpp.grad(image)

            # Initial and final images are fixed, with zero gradient
            if i == 0 or i == len(images) - 1:
                image.gradient[:] = 0.0

        result = minimize(
            total_energy,
            x0=images.coords(),
            method="L-BFGS-B",
            jac=derivative,
            args=(images, idpp, Config.n_cores, False),
            options={"gtol": 0.01},
        )

        logger.info(f"IDPP minimisation successful: {result.success}")

        self.images.set_coords(result.x)
        return None

    def _max_atom_distance_between_images(
        self, idxs: Optional[Sequence[int]] = None
    ) -> Distance:
        """
        Calculate the maximum atom-atom distance between two consecutive images
        """
        if idxs is None:  # Use all pairwise distances
            idxs = np.arange(self.images[0].n_atoms)
        else:
            idxs = np.array(idxs)

        overall_max_distance = -np.inf

        for i in range(len(self.images) // 2):
            k = 2 * i
            x_i = self.images[k].coordinates
            x_j = self.images[k + 1].coordinates

            max_distance = np.max(np.linalg.norm(x_i - x_j, axis=1)[idxs])
            if max_distance > overall_max_distance:
                overall_max_distance = max_distance

        return overall_max_distance

    @property
    def max_atom_distance_between_images(self) -> Distance:
        return self._max_atom_distance_between_images(idxs=None)

    @staticmethod
    def _raise_exception_if_any(kwargs: dict) -> None:
        if len(kwargs) == 0:
            return
        elif any(
            arg in kwargs
            for arg in ("initial_species", "final_species", "num")
        ):
            raise ValueError(
                "Cannot construct a NEB. Please use NEB.from_endpoints()"
            )
        elif "species_list" in kwargs:
            raise ValueError(
                "Cannot construct a NEB from a species list. Please use NEB.from_list()"
            )
        else:
            raise ValueError("Unrecognised keyword argument")
