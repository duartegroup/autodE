"""
The theory behind this original NEB implementation is taken from
Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)
"""
from typing import Optional, Sequence
from autode.log import logger
from autode.calculations import Calculation
from autode.wrappers.methods import Method
from autode.input_output import xyz_file_to_molecules
from autode.path import Path
from autode.utils import work_in
from autode.config import Config
from autode.neb.idpp import IDPP
from scipy.optimize import minimize
from autode.values import Distance, PotentialEnergy
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

blues = plt.get_cmap("Blues")


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
        molecule=image.species,
        method=est_method,
        keywords=est_method.keywords.grad,
        n_cores=n_cores,
    )

    @work_in(image.name)
    def run():
        calc.run()
        image.grad = image.species.gradient.flatten()
        image.energy = image.species.energy
        return None

    run()
    return image


def _idpp_energy_gradient(
    image: "autode.neb.original.Image",
    idpp: "autode.neb.idpp.IDPP",
    n_cores: int,
) -> "autode.neb.original.Image":
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
    image.grad = idpp.grad(image)

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

    # Run an energy + gradient evaluation in parallel across all images
    with Pool(processes=n_cores) as pool:
        results = [
            pool.apply_async(
                func=energy_gradient, args=(images[i], method, n_cores_pp)
            )
            for i in range(1, len(images) - 1)
        ]

        images[1:-1] = [result.get(timeout=None) for result in results]

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
    forces = np.array(images[0].grad)

    # No need to calculate gradient as should already be there from energy eval
    for i in range(1, len(images) - 1):
        force = images[i].get_force(im_l=images[i - 1], im_r=images[i + 1])
        forces = np.append(forces, force)

    # Final zero set of forces
    forces = np.append(forces, images[-1].grad)

    # dV/dx is negative of the force
    logger.info(f"|F| = {np.linalg.norm(forces):.4f} Ha Å-1")
    return -forces


class Image:
    def __init__(self, name: str, k: float):
        """
        Image in a NEB

        Arguments:
            name (str):
        """
        self.name = name

        # Current optimisation iteration of this image
        self.iteration = 0

        # Force constant in Eh/Å^2
        self.k = k

        self.species: Optional["autode.species.Species"] = None
        self.energy: Optional["autode.values.Energy"] = None
        self.grad: Optional[np.ndarray] = None

    def _tau_xl_x_xr(
        self,
        im_l: "autode.neb.original.Image",
        im_r: "autode.neb.original.Image",
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
            image.species.coordinates.flatten() for image in (im_l, self, im_r)
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
        im_l: "autode.neb.original.Image",
        im_r: "autode.neb.original.Image",
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
        # τ,  x_i-1,  x_i,   x_i+1
        hat_tau, x_l, x, x_r = self._tau_xl_x_xr(im_l, im_r)

        # F_i^s||
        f_parallel = (
            np.linalg.norm(x_r - x) * im_r.k - np.linalg.norm(x - x_l) * im_l.k
        ) * hat_tau

        # ∇V(x)_i|_|_ = ∇V(x)_i - (∇V(x)_i•τ) τ
        grad_perp = self.grad - np.dot(self.grad, hat_tau) * hat_tau

        # F_i = F_i^s|| -  ∇V(x)_i|_|_
        return f_parallel - grad_perp


class Images(Path):
    def __init__(self, num, init_k, min_k=None, max_k=None):
        """
        Set of images joined by harmonic springs with force constant k

        -----------------------------------------------------------------------
        Arguments:

            num (int): Number of images

            init_k (float): Initial force constant (Ha Å^-2)

            min_k (None | float): Minimum value of k

            max_k (None | float): Maximum value of k
        """
        super().__init__(*(Image(name=str(i), k=init_k) for i in range(num)))

        self.min_k = init_k / 10 if min_k is None else float(min_k)
        self.max_k = 2 * init_k if max_k is None else float(max_k)

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
                    image.k = self.max_k - delta_k * (
                        (e_max - image.energy) / (e_max - e_ref)
                    )
        return None

    def plot_energies(
        self, save=False, name="None", color=None, xlabel="NEB coordinate"
    ):
        """Plot the NEB surface"""
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

            coords = np.append(coords, image.species.coordinates.flatten())
        return coords

    def set_coords(self, coords):
        """
        Set the flat array of coordinates to the species in the images

        -----------------------------------------------------------------------
        Arguments:
            coords (np.ndarray): shape (num x n x 3,)
        """

        n_atoms = self[0].species.n_atoms
        coords = coords.reshape((len(self), n_atoms, 3))

        for i, image in enumerate(self):
            image.species.coordinates = coords[i]

        return None

    def copy(self) -> "Images":
        return deepcopy(self)


class NEB:

    _images_type = Images

    def __init__(
        self,
        initial_species: Optional["autode.Species"] = None,
        final_species: Optional["autode.Species"] = None,
        num: int = 8,
        species_list: Optional[list] = None,
        k: float = 0.1,
    ):
        """
        Nudged elastic band

        -----------------------------------------------------------------------
        Arguments:
            initial_species (autode.species.Species):

            final_species (autode.species.Species):

            num (int): Number of images in the NEB

            species_list (list(autode.species.Species)): Intermediate images
                         along the NEB
        """
        self._init_k = k

        if species_list is not None:
            self.images = Images(num=len(species_list), init_k=k)

            for image, species in zip(self.images, species_list):
                image.species = species
                image.energy = species.energy

        else:
            self.images = Images(num=num, init_k=k)
            self._init_from_end_points(initial_species, final_species)

        logger.info(f"Initialised a NEB with {len(self.images)} images")

    @property
    def init_k(self) -> float:
        """Initial force constant used to in this NEB. Units of Ha Å^-1"""
        return self._init_k

    @classmethod
    def from_file(
        cls,
        filename: str,
        k: Optional[float] = None,
    ) -> "NEB":
        """
        Create a nudged elastic band from a .xyz file containing multiple
        images.
        """

        molecules = xyz_file_to_molecules(filename)
        if k is None and all(m.energy is not None for m in molecules):
            logger.info(
                "Have a set of energies from file. Can adaptively "
                "choose a sensible force constant (k)"
            )

            max_de = max(
                abs(molecules[i].energy - molecules[i + 1].energy)
                for i in range(len(molecules) - 1)
            )

            # TODO: test reasonableness of this function...
            # use a shifted tanh to interpolate in [0.005, 0.2005]
            k = (
                0.1 * (np.tanh((max_de.to("kcal mol-1") - 40) / 20) + 1)
                + 0.005
            )

        if k is None:  # choose a sensible default
            k = 0.1

        logger.info(f"Using k = {k:.6f} Ha Å^-1 as the NEB force constant")
        return cls(species_list=molecules, k=k)

    def _minimise(self, method, n_cores, etol, max_n=30) -> None:
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

    def contains_peak(self) -> bool:
        """Does this nudged elastic band calculation contain an energy peak?"""
        return self.images.contains_peak

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
        init_k = self.images[0].k
        _list = []

        for i, left_image in enumerate(self.images[:-1]):
            right_image = self.images[i + 1]

            left_species, right_species = (
                left_image.species,
                right_image.species,
            )
            sub_neb = NEB(species_list=[left_species, right_species])

            n = 2

            while (
                sub_neb._max_atom_distance_between_images(distance_idxs)
                > max_delta
            ):

                sub_neb.images = self._images_type(num=n, init_k=init_k)
                sub_neb.images[0].species = left_species
                sub_neb.images[-1].species = right_species
                sub_neb.interpolate_geometries()

                try:
                    sub_neb.idpp_relax()
                except RuntimeError:
                    logger.warning("Failed to IDPP relax the interpolated NEB")

                n += 1

            for image in sub_neb.images[:-1]:  # add all apart from the last
                _list.append(image.species)

        _list.append(self._species_at(-1))  # end with the last
        self.images = self._images_type(num=len(_list), init_k=init_k)

        for i, image in enumerate(self.images):
            image.species = _list[i]

        logger.info(
            f"Partition successful – now have {len(self.images)} " f"images"
        )
        return None

    def _species_at(self, idx: int) -> "autode.species.Species":
        """Return the species associated with a particular image index"""
        return self.images[idx].species

    def print_geometries(self, name="neb") -> None:
        return self.images.print_geometries(name)

    def interpolate_geometries(self) -> None:
        """Generate simple interpolated coordinates for these set of images
        in Cartesian coordinates"""
        n = len(self.images)

        # Interpolate images between the starting point i=0 and end point i=n-1
        for i in range(1, n - 1):

            # Use a copy of the starting point for atoms, charge etc.
            self.images[i].species = self.images[0].species.copy()

            # For all the atoms in the species translate an amount so the
            # spacing is even between the initial and final points
            for j, atom in enumerate(self.images[i].species.atoms):

                # Shift vector is final minus current
                shift = self.images[-1].species.atoms[j].coord - atom.coord
                # then an equal spacing is the i-th point in the grid
                atom.translate(vec=shift * (i / n))

        return None

    @work_in("neb")
    def calculate(
        self,
        method: "autode.wrappers.methods.Method",
        n_cores: int,
        name_prefix: str = "",
        etol_per_image: PotentialEnergy = PotentialEnergy(
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
        self.print_geometries(name=f"{name_prefix}neb_init")

        # Calculate energy on the first and final points
        for idx in [0, -1]:
            energy_gradient(self.images[idx], method=method, n_cores=n_cores)
            # Zero the forces so the end points don't move
            self.images[idx].grad = np.zeros(shape=self.images[idx].grad.shape)

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

    def get_species_saddle_point(self) -> "autode.species.Species":
        """Find a TS guess species for this NEB: highest energy saddle point"""
        if self.images.peak_idx is None:
            logger.warning("Found no peaks in the NEB")
            return None

        return self.images[self.images.peak_idx].species

    def _init_from_end_points(self, initial, final) -> None:
        """Initialise from the start and finish points of the NEB"""

        self.images[0].species = initial
        self.images[-1].species = final
        self.interpolate_geometries()
        self.idpp_relax()

        return None

    def idpp_relax(self) -> None:
        """
        Relax the NEB using the image dependent pair potential

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`IDPP <autode.neb.idpp.IDPP.__init__>`
        """
        logger.info(f"Minimising NEB with IDPP potential")
        images = self.images.copy()
        idpp = IDPP(images)

        images.min_k = images.max_k = 0.1

        for i, image in enumerate(images):
            image.energy = idpp(image)
            image.grad = idpp.grad(image)

            # Initial and final images are fixed, with zero gradient
            if i == 0 or i == len(images) - 1:
                image.grad[:] = 0.0

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
            idxs = np.arange(self.images[0].species.n_atoms)
        else:
            idxs = np.array(idxs)

        overall_max_distance = -np.inf

        for i in range(len(self.images) // 2):
            k = 2 * i
            x_i = self._species_at(k).coordinates
            x_j = self._species_at(k + 1).coordinates

            max_distance = np.max(np.linalg.norm(x_i - x_j, axis=1)[idxs])
            if max_distance > overall_max_distance:
                overall_max_distance = max_distance

        return overall_max_distance

    @property
    def max_atom_distance_between_images(self) -> Distance:
        return self._max_atom_distance_between_images(idxs=None)
