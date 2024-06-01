"""
Base classes for implementing all bracketing methods
that require a pair of images
"""
import itertools
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING, Union, Iterator
from enum import Enum

from autode.values import Distance, PotentialEnergy, Gradient
from autode.geom import get_rot_mat_kabsch
from autode.methods import get_lmethod
from autode.neb import CINEB
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.opt.optimisers.utils import Polynomial2PointFit
from autode.opt.optimisers.base import OptimiserHistory, print_geometries_from
from autode.plotting import plot_bracket_method_energy_profile
from autode.utils import work_in_tmp_dir, ProcessPool
from autode.log import logger

if TYPE_CHECKING:
    from autode.species import Species
    from autode.wrappers.methods import Method
    from autode.hessians import Hessian


def _calculate_engrad_for_species(
    species: "Species",
    method: "Method",
    n_cores: int,
) -> Tuple[PotentialEnergy, Gradient]:
    """
    Convenience function for calculating the energy/gradient
    for a molecule; removes all input and output files after
    the calculation is finished

    Returns:
        (tuple[PotentialEnergy, Gradient]): Energy and gradient as tuple
    """
    from autode.calculations import Calculation

    engrad_calc = Calculation(
        name=f"{species.name}_engrad",
        molecule=species,
        method=method,
        keywords=method.keywords.grad,
        n_cores=n_cores,
    )
    engrad_calc.run()
    engrad_calc.clean_up(force=True, everything=True)
    assert species.energy and species.gradient is not None, "Calc must be ok"

    return species.energy, species.gradient


@work_in_tmp_dir()
def _calculate_hessian_for_species(
    species: "Species",
    method: "Method",
    n_cores: int,
) -> "Hessian":
    """
    Convenience function for calculating the Hessian for a
    molecule; removes all input and output files after
    the calculation is finished

    Returns:
        (Hessian): Hessian matrix
    """
    from autode.calculations import Calculation

    species = species.new_species()

    hess_calc = Calculation(
        name=f"{species.name}_hess",
        molecule=species,
        method=method,
        keywords=method.keywords.hess,
        n_cores=n_cores,
    )
    hess_calc.run()
    hess_calc.clean_up(force=True, everything=True)
    assert species.hessian is not None, "Calc must be ok"

    return species.hessian


class BaseImagePair(ABC):
    """
    Base class for a pair of images (e.g., reactant and product) of
    the same species. The images are called 'left' and 'right' to
    distinguish them, but there is no requirement for one to be
    reactant or product. Calculations can be performed on both sides
    parallely
    """

    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
    ):
        """
        Initialize the image pair, does not set methods/n_cores

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
        """
        from autode.species.species import Species

        assert isinstance(left_image, Species)
        assert isinstance(right_image, Species)
        self._left_image = left_image.new_species(name="left_image")
        self._right_image = right_image.new_species(name="right_image")
        self._sanity_check()
        self._align_species()

        # for calculation
        self._method = None
        self._hess_method = None
        self._n_cores = None
        self._hessian_update_types = [BofillUpdate]

        self._left_history = OptimiserHistory()
        self._right_history = OptimiserHistory()
        # push the first coordinates into history
        self.left_coords = CartesianCoordinates(self._left_image.coordinates)
        self.right_coords = CartesianCoordinates(self._right_image.coordinates)

    def _sanity_check(self) -> None:
        """
        Check if the two supplied images have the same solvent,
        charge, multiplicity and the same atoms in the same order
        """

        if self._left_image.n_atoms != self._right_image.n_atoms:
            raise ValueError(
                "The initial_species and final_species must "
                "have the same number of atoms!"
            )

        if (
            self._left_image.charge != self._right_image.charge
            or self._left_image.mult != self._right_image.mult
            or self._left_image.solvent != self._right_image.solvent
        ):
            raise ValueError(
                "Charge/multiplicity/solvent of initial_species "
                "and final_species supplied are not the same"
            )

        for idx in range(len(self._left_image.atoms)):
            if (
                self._left_image.atoms[idx].label
                != self._right_image.atoms[idx].label
            ):
                raise ValueError(
                    "The order of atoms in initial_species "
                    "and final_species must be the same. The "
                    f"atom at position {idx} is different in"
                    "the two species"
                )

        return None

    def _align_species(self) -> None:
        """
        Translates both molecules to origin and then performs
        a Kabsch rotation to orient the molecules as close as
        possible against each other
        """
        # first translate the molecules to the origin
        logger.info(
            "Translating initial_species (reactant) "
            "and final_species (product) to origin"
        )
        p_mat = self._left_image.coordinates.copy()
        p_mat -= np.average(p_mat, axis=0)
        self._left_image.coordinates = p_mat

        q_mat = self._right_image.coordinates.copy()
        q_mat -= np.average(q_mat, axis=0)
        self._right_image.coordinates = q_mat

        logger.info(
            "Rotating initial_species (reactant) "
            "to align with final_species (product) "
            "as much as possible"
        )
        rot_mat = get_rot_mat_kabsch(p_mat, q_mat)
        rotated_p_mat = np.dot(rot_mat, p_mat.T).T
        self._left_image.coordinates = rotated_p_mat

    def initialise_trj(
        self,
        left_history_name: str = "left_history_save.zip",
        right_history_name: str = "right_history_save.zip",
    ) -> None:
        """
        Initialise the trajectory save files (history of coordinates on
        left and right images)

        Args:
            left_history_name: Name of savefile for left history
            right_history_name: Name of savefile for right history
        """
        self._left_history.open(left_history_name)
        self._right_history.open(right_history_name)

    def close_trj(self):
        """
        Put all coordinates in memory onto disk in the trajectory
        save files, and close the trajectories
        """
        self._left_history.close()
        self._right_history.close()

    def set_method_and_n_cores(
        self,
        method: "Method",
        n_cores: int,
        hess_method: Optional["Method"] = None,
    ) -> None:
        """
        Sets the methods for en/grad calculation, and the total
        number of cores used for any calculation in this image pair.
        Optionally, also set the method for hessian calculation; if
        not set, the available lmethod will be used.

        Args:
            method (Method): Method used for calculating energy/gradient
            n_cores (int): Number of cores available
            hess_method (Method|None): Method used for calculating
                                       Hessian (optional)
        """
        from autode.wrappers.methods import Method

        if not isinstance(method, Method):
            raise TypeError(
                f"The method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(method)} was supplied."
            )
        self._method = method

        if hess_method is None:
            hess_method = get_lmethod()

        if not isinstance(hess_method, Method):
            raise TypeError(
                f"The hessian method needs to be of type autode."
                f"wrappers.method.Method, But {type(hess_method)}"
                f"was supplied"
            )
        self._hess_method = hess_method

        self._n_cores = int(n_cores)
        return None

    @property
    def n_atoms(self) -> int:
        """Number of atoms"""
        return self._left_image.n_atoms

    @property
    def total_iters(self) -> int:
        """Total number of iterations done on this image pair"""
        return len(self._left_history) + len(self._right_history) - 2

    @property
    def left_coords(self) -> CartesianCoordinates:
        """The coordinates of the left image"""
        assert isinstance(self._left_history[-1], CartesianCoordinates)
        return self._left_history[-1]

    @left_coords.setter
    def left_coords(self, value: CartesianCoordinates):
        """
        Sets the coordinates of the left image, also updates
        the coordinates of the species

        Args:
            value (CartesianCoordinates|None): new set of coordinates

        Raises:
            (TypeError): If input is not of type CartesianCoordinates
            (ValueError): If input does not have correct shape
        """
        if value.shape[0] != 3 * self.n_atoms:
            raise ValueError(f"Must have {self.n_atoms * 3} entries")

        if isinstance(value, CartesianCoordinates):
            self._left_history.add(value.copy())
        else:
            raise TypeError

        self._left_image.coordinates = self.left_coords

    @property
    def right_coords(self) -> CartesianCoordinates:
        """The coordinates of the right image"""
        assert isinstance(self._right_history[-1], CartesianCoordinates)
        return self._right_history[-1]

    @right_coords.setter
    def right_coords(self, value: CartesianCoordinates):
        """
        Sets the coordinates of the right image, also updates
        the coordinates of the species

        Args:
            value (CartesianCoordinates|None): new set of coordinates

        Raises:
            (TypeError): If input is not of type CartesianCoordinates
            (ValueError): If input does not have correct shape
        """
        if value.shape[0] != 3 * self.n_atoms:
            raise ValueError(f"Must have {self.n_atoms * 3} entries")

        if isinstance(value, CartesianCoordinates):
            self._right_history.add(value.copy())
        else:
            raise TypeError

        self._right_image.coordinates = self.right_coords

    @property
    @abstractmethod
    def ts_guess(self) -> Optional["Species"]:
        """TS guess species for this image-pair"""

    @property
    @abstractmethod
    def dist_vec(self) -> np.ndarray:
        """Distance vector defined from left to right image"""

    @property
    @abstractmethod
    def dist(self) -> Distance:
        """Distance defined between two images in the image-pair"""

    @property
    @abstractmethod
    def has_jumped_over_barrier(self) -> bool:
        """Whether one image has jumped over the barrier on the other side"""

    def update_both_img_engrad(self):
        """
        Update the energy/gradient for both images, with parallel processing
        """
        assert self._method is not None
        assert self._n_cores is not None
        n_cores_per_pp = self._n_cores // 2 if self._n_cores > 1 else 1
        n_procs = 1 if self._n_cores < 2 else 2
        with ProcessPool(max_workers=n_procs) as pool:
            jobs = [
                pool.submit(
                    _calculate_engrad_for_species,
                    species=img,
                    method=self._method,
                    n_cores=n_cores_per_pp,
                )
                for img in [self._left_image, self._right_image]
            ]
            left_engrad, right_engrad = [job.result() for job in jobs]

        self.left_coords.e = left_engrad[0]
        self.left_coords.update_g_from_cart_g(left_engrad[1])
        self.right_coords.e = right_engrad[0]
        self.right_coords.update_g_from_cart_g(right_engrad[1])
        return None

    def update_both_img_hessian_by_calc(self):
        """
        Update the molecular hessian of both images by calculation
        """
        # TODO: refactor into ll_hessian code
        assert self._hess_method is not None
        assert self._n_cores is not None
        n_cores_per_pp = self._n_cores // 2 if self._n_cores > 1 else 1
        n_procs = 1 if self._n_cores < 2 else 2
        with ProcessPool(max_workers=n_procs) as pool:
            jobs = [
                pool.submit(
                    _calculate_hessian_for_species,
                    species=img,
                    method=self._hess_method,
                    n_cores=n_cores_per_pp,
                )
                for img in [self._left_image, self._right_image]
            ]
            left_hess, right_hess = [job.result() for job in jobs]

        self.left_coords.update_h_from_cart_h(left_hess)
        self.right_coords.update_h_from_cart_h(right_hess)
        return None

    def update_both_img_hessian_by_formula(self):
        """
        Update the molecular hessian for both images by update formula
        """
        for history in [self._left_history, self._right_history]:
            history.final.update_h_from_old_h(
                history.penultimate, self._hessian_update_types
            )

        return None


class EuclideanImagePair(BaseImagePair, ABC):
    """
    Image-pair that defines the distance between the images as
    the Euclidean distance. It can also run CI-NEB calculation
    from the final two points added to the image-pair, and
    plot the energies of the total path
    """

    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
    ):
        super().__init__(left_image=left_image, right_image=right_image)

        # for storing results from CINEB
        self._cineb_coords: Optional[CartesianCoordinates] = None

    @property
    def dist_vec(self) -> np.ndarray:
        """
        Distance vector in cartesian coordinates, it is defined here to
        go from right to left image (i.e. right -> left)
        """
        return np.array(
            self.left_coords.to("cart") - self.right_coords.to("cart")
        )

    @property
    def dist(self) -> Distance:
        """
        Euclidean distance between the images in image-pair

        Returns:
            (Distance): Distance in Angstrom
        """
        return Distance(np.linalg.norm(self.dist_vec), units="ang")

    @property
    def has_jumped_over_barrier(self) -> bool:
        """
        A quick test of whether the images are still separated by a barrier,
        implemented via fitting a cubic polynomial along the linear path
        connecting the two images and checking for a peak. This is only an
        approximation.
        """
        assert self.left_coords is not None and self.right_coords is not None
        assert self.left_coords.e and self.right_coords.e
        assert (
            self.left_coords.g is not None and self.right_coords.g is not None
        )
        cubic_poly = Polynomial2PointFit.cubic_fit(
            self.left_coords, self.right_coords
        )

        # NOTE: Interpolation seems reasonable upto ~1.2 Angstrom. If distance
        # is larger, detecting peak is impossible without calculating energies
        # so we assume there is a barrier between the images
        if self.dist > Distance(1.2, "ang"):
            return False
        else:
            return cubic_poly.get_extremum(0.0, 1.0, get_max=True) is None

    def run_cineb_from_end_points(self) -> None:
        """
        Runs a CI-NEB calculation from the end-points of the image-pair
        and then stores the coordinates of the peak point obtained
        from the CI-NEB run

        Returns:
            (CartesianCoordinates): Coordinates of the peak species obtained
                                    from the CI-NEB run
        """
        assert self._method is not None, "Methods must be set"
        assert self._n_cores is not None, "Number of cores must be set"

        cineb = CINEB.from_end_points(
            self._left_image, self._right_image, num=3
        )
        cineb.calculate(method=self._method, n_cores=self._n_cores)

        if not cineb.images.contains_peak:
            logger.error("CI-NEB failed to find the peak")
            return None

        peak = cineb.images[cineb.images.peak_idx]  # type: ignore
        ci_coords = CartesianCoordinates(peak.coordinates)
        ci_coords.e = peak.energy
        ci_coords.update_g_from_cart_g(peak.gradient)

        self._cineb_coords = ci_coords
        return None

    @property
    def _total_history(self) -> Iterator[CartesianCoordinates]:
        """
        The total history of the image-pair, including any CI run
        from the endpoints
        """
        cineb_coords = []
        if self._cineb_coords is not None:
            cineb_coords.append(self._cineb_coords)
        return itertools.chain(
            self._left_history, cineb_coords, reversed(self._right_history)
        )

    def print_geometries(
        self,
        init_trj_filename: str,
        final_trj_filename: str,
        total_trj_filename: str,
    ) -> None:
        """
        Write trajectories as *.xyz files, one for the initial species,
        one for final species, and one for the whole trajectory, including
        any CI-NEB run from the final end points
        """
        if self.total_iters < 2:
            logger.warning("Cannot write trajectory, not enough points")
            return None

        print_geometries_from(
            self._left_history,
            species=self._left_image,
            filename=init_trj_filename,
        )
        print_geometries_from(
            self._right_history,
            species=self._right_image,
            filename=final_trj_filename,
        )
        print_geometries_from(
            self._total_history,
            species=self._left_image,
            filename=total_trj_filename,
        )

        return None

    def plot_energies(
        self,
        filename: str,
        distance_metric: str,
    ) -> None:
        """
        Plots the energies of the image-pair, including any CI-NEB
        calculation done at the end. The distance metric argument
        determines how the x-axis values are plotted and their
        meaning (Described in more detail in BaseBracketMethod)

        Args:
            filename (str): name of the plot file to save
            distance_metric (str): "relative" or "from_start" or "index"

        See Also:
            :py:meth:`BaseBracketMethod <autode.bracket.base.BaseBracketMethod.plot_energies>`
        """

        class Metrics(Enum):
            relative = 1
            from_start = 2
            index = 3

        metric = Metrics[distance_metric]

        if self.total_iters < 2:
            logger.warning("Cannot plot energies, not enough points")
            return None

        all_energies = [coord.e for coord in self._total_history]
        if any(en is None for en in all_energies):
            logger.error(
                "One or more coordinates do not have associated"
                " energies, unable to produce energy plot!"
            )
            return None

        num_left_points = len(self._left_history)
        num_right_points = len(self._right_history)
        first_point = self._left_history[0]
        points: list = []  # list of tuples

        lowest_en = min(all_energies)  # type: ignore

        last_coord = None
        for idx, coord in enumerate(self._total_history):
            en = coord.e - lowest_en  # type: ignore
            if metric == Metrics.relative:
                if idx == 0:
                    x = 0
                else:
                    x = np.linalg.norm(coord - last_coord)
                    x += points[idx - 1][0]  # add previous distance
            elif metric == Metrics.from_start:
                x = np.linalg.norm(coord - first_point)
            else:  # metric == Metrics.index:
                x = idx
            points.append((x, en))
            last_coord = coord

        left_points = points[:num_left_points]
        if self._cineb_coords is not None:
            cineb_point = points[num_left_points]
        else:
            cineb_point = None
        right_points = points[-num_right_points:]
        if distance_metric == "relative":
            x_axis_title = "Change in Euclidean Distance (Å)"
        elif distance_metric == "from_start":
            x_axis_title = "Euclidean Distance from Reactant Structure (Å)"
        else:
            x_axis_title = "Point in Reaction Path"

        plot_bracket_method_energy_profile(
            filename, left_points, cineb_point, right_points, x_axis_title
        )
