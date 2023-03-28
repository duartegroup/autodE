"""
Base classes for implementing all bracketing methods
that require a pair of images
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

from autode.values import PotentialEnergy, Gradient, Distance
from autode.hessians import Hessian
from autode.geom import get_rot_mat_kabsch
from autode.neb import CINEB
from autode.opt.coordinates import CartesianCoordinates, OptCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.opt.optimisers.base import _OptimiserHistory
from autode.utils import work_in_tmp_dir, ProcessPool
from autode.exceptions import CalculationException
from autode.plotting import plot_bracket_method_energy_profile
from autode.log import logger

if TYPE_CHECKING:
    from autode.species import Species
    from autode.wrappers.methods import Method


def _calculate_engrad_for_species(
    species: "Species",
    method: "Method",
    n_cores: int,
) -> Tuple[PotentialEnergy, Gradient]:
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
    if species.energy is None or species.gradient is None:
        raise CalculationException("Energy/gradient calculation failed")

    return species.energy, species.gradient


def _calculate_energy_for_species(
    species: "Species",
    method: "Method",
    n_cores: int,
) -> PotentialEnergy:
    from autode.calculations import Calculation

    sp_calc = Calculation(
        name=f"{species.name}_sp",
        molecule=species,
        method=method,
        keywords=method.keywords.sp,
        n_cores=n_cores,
    )
    sp_calc.run()
    sp_calc.clean_up(force=True, everything=True)
    if species.energy is None:
        raise CalculationException("Energy calculation failed")

    return species.energy


@work_in_tmp_dir()
def _calculate_hessian_for_species(
    species: "Species",
    method: "Method",
    n_cores: int,
) -> Hessian:
    """
    Convenience function for calculating the Hessian for a
    molecule; removes all input and output files for the
    calculation
    """
    from autode.calculations import Calculation

    hess_calc = Calculation(
        name=f"{species.name}_hess",
        molecule=species,
        method=method,
        keywords=method.keywords.hess,
        n_cores=n_cores,
    )
    hess_calc.run()
    hess_calc.clean_up(force=True, everything=True)
    if species.hessian is None:
        raise CalculationException("Hessian calculation failed")

    return species.hessian


def _remove_file_with_warning(filename: str) -> None:
    """Removes a file if it is present, and warns the user"""
    if os.path.isfile(filename):
        logger.warning(f"{filename} exists, overwriting...")
        os.remove(filename)
    return None


class ImgPairSideError(ValueError):
    """
    Error if side is neither 'left' nor 'right', used only for internal
    consistency, as the functions should not be called by user
    """

    def __init__(self):
        super().__init__("Side supplied must be either 'left' or 'right'")


class BaseImagePair(ABC):
    """
    Base class for a pair of images (e.g., reactant and product) of
    the same species. The images are called 'left' and 'right' to
    distinguish them, but there is no requirement for one to be
    reactant or product.
    """

    def __init__(
        self,
        left_image: "Species",
        right_image: "Species",
    ):
        """
        Initialize the image pair, does not set methods/n_cores or
        hessian update types!

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

        # separate methods for engrad and hessian calc
        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None

        # Bofill has no conditions, so kept as default
        self._hessian_update_types = [BofillUpdate]

        self._left_history = _OptimiserHistory()
        self._right_history = _OptimiserHistory()
        # push the first coordinates into history
        self.left_coord = CartesianCoordinates(
            self._left_image.coordinates.to("ang")
        )
        self.right_coord = CartesianCoordinates(
            self._right_image.coordinates.to("ang")
        )
        # todo replace type hints with optcoordiantes

        # Store coords from CI-NEB
        self._cineb_coords = None

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

    def set_method_and_n_cores(
        self,
        engrad_method: "Method",
        n_cores: int,
        hess_method: Optional["Method"] = None,
    ) -> None:
        """
        Sets the methods for engrad and hessian calculation, and the
        total number of cores used for any calculation in this image pair

        Args:
            engrad_method (Method):
            n_cores (int):
            hess_method (Method|None):
        """
        from autode.wrappers.methods import Method

        if not isinstance(engrad_method, Method):
            raise TypeError(
                f"The engrad_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(engrad_method)} was supplied."
            )
        self._engrad_method = engrad_method

        if hess_method is None:
            pass
        elif not isinstance(hess_method, Method):
            raise TypeError(
                f"The hess_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(hess_method)} was supplied."
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
    def left_coord(self) -> Optional[OptCoordinates]:
        """The coordinates of the left image"""
        if len(self._left_history) == 0:
            return None
        return self._left_history[-1]

    @left_coord.setter
    def left_coord(self, value: Optional[OptCoordinates]):
        """
        Sets the coordinates of the left image, also updates
        the coordinates of the species

        Args:
            value (OptCoordinates|None): new set of coordinates

        Raises:
            (TypeError): If input is not of type CartesianCoordinates
            (ValueError): If input does not have correct shape
        """
        if value is None:
            return
        if value.shape[0] != 3 * self.n_atoms:
            raise ValueError(f"Must have {self.n_atoms * 3} entries")

        if isinstance(value, OptCoordinates):
            self._left_history.append(value.copy())
        else:
            raise TypeError

        self._left_image.coordinates = self.left_coord.to("cart")
        # todo should we remove old hessians that are not needed to free mem?

    @property
    def right_coord(self) -> Optional[OptCoordinates]:
        """The coordinates of the right image"""
        if len(self._right_history) == 0:
            return None
        return self._right_history[-1]

    @right_coord.setter
    def right_coord(self, value: Optional[OptCoordinates]):
        """
        Sets the coordinates of the right image, also updates
        the coordinates of the species

        Args:
            value (OptCoordinates|None): new set of coordinates

        Raises:
            (TypeError): If input is not of type CartesianCoordinates
            (ValueError): If input does not have correct shape
        """
        if value is None:
            return
        if value.shape[0] != 3 * self.n_atoms:
            raise ValueError(f"Must have {self.n_atoms * 3} entries")

        if isinstance(value, OptCoordinates):
            self._right_history.append(value.copy())
        else:
            raise TypeError

        self._right_image.coordinates = self.right_coord.to("cart")

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
    def dist(self):
        """Distance defined between two images in the image-pair"""

    @property
    def _total_history(self) -> _OptimiserHistory:
        """
        The total history of the image-pair, including any CI run
        from the endpoints
        """
        history = _OptimiserHistory()
        history.extend(self._left_history)
        if self._cineb_coords is not None:
            history.append(self._cineb_coords)
        history.extend(self._right_history[::-1])  # reverse order
        return history

    def run_cineb_from_end_points(self) -> None:
        """
        Runs a CI-NEB calculation from the end-points of the image-pair
        and then stores the coordinates of the peak point obtained
        from the CI-NEB run

        Returns:
            (CartesianCoordinates): Coordinates of the peak species obtained
                                    from the CI-NEB run
        """
        assert self._engrad_method is not None, "Methods must be set"
        assert self._n_cores is not None, "Number of cores must be set"

        if self.dist > 2.0:
            logger.warning(
                "The distance between the images in image-pair is"
                "quite large, bracketing method may not have "
                "converged completely."
            )

        cineb = CINEB.from_end_points(
            self._left_image, self._right_image, num=3
        )
        cineb.calculate(method=self._engrad_method, n_cores=self._n_cores)

        ci_coords = CartesianCoordinates(
            cineb.peak_species.coordinates.to("ang")
        )
        ci_coords.e = cineb.peak_species.energy
        ci_coords.update_g_from_cart_g(cineb.peak_species.gradient)

        self._cineb_coords = ci_coords

        return None

    def write_trajectories(
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

        _remove_file_with_warning(init_trj_filename)
        self._write_trj_from_history(self._left_history, init_trj_filename)

        _remove_file_with_warning(final_trj_filename)
        self._write_trj_from_history(self._right_history, final_trj_filename)

        _remove_file_with_warning(total_trj_filename)
        self._write_trj_from_history(self._total_history, total_trj_filename)

        return None

    def _write_trj_from_history(
        self,
        history: _OptimiserHistory,
        filename: str,
    ) -> None:
        """
        Convenience function to write an *.xyz trajectory from a
        coordinate history, using the Species

        Args:
            history (_OptimiserHistory): History of coordinate objects
            filename (str): Name of the file to store the coordinates
        """
        tmp_spc = self._left_image.copy()
        for coord in history:
            tmp_spc.coordinates = coord
            tmp_spc.energy = coord.e
            tmp_spc.print_xyz_file(filename=filename, append=True)

        return None

    def plot_energies(
        self,
        filename: str,
        distance_metric: str,
    ):
        """
        Plots the energies of the image-pair, including any CI-NEB
        calculation done at the end. The distance metric argument
        determines how the x-axis values are plotted and their
        meaning (Described in more detail in BaseBracketMethod)

        Args:
            filename: name of the plot file to save
            distance_metric: "relative" or "from_start" or "index"

        See Also:
            :py:meth:`BaseBracketMethod <autode.bracket.base.BaseBracketMethod.plot_energies>`
        """
        if self.total_iters < 2:
            logger.warning("Cannot plot energies, not enough points")

        if distance_metric not in ["relative", "from_start", "index"]:
            raise ValueError(
                "The distance metric must be 'relative', 'from_start' or 'index'"
            )

        num_left_points = len(self._left_history)
        num_right_points = len(self._right_history)
        first_point = self._left_history[0]
        points = []  # list of tuples

        lowest_en = min(coord.e for coord in self._total_history)

        for idx, coord in enumerate(self._total_history):
            en = coord.e
            if en is None:
                en = np.nan
            else:
                en = en - lowest_en
                en = float(en.to("kcalmol"))
            if distance_metric == "relative":
                if idx == 0:
                    x = 0
                else:
                    x = np.linalg.norm(coord - self._total_history[idx - 1])
                    x += points[idx - 1]
            elif distance_metric == "from_start":
                x = np.linalg.norm(coord - first_point)
            else:
                x = idx
            points.append((x, en))

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


class ImagePair(BaseImagePair, ABC):
    """
    Image-pair that defines the distance between the images as
    the Euclidean distance. Single-point, en/grad, and hessian
    calculation can be run, and hessian update can be done
    using gradient information
    """

    @property
    def dist_vec(self) -> np.ndarray:
        """Distance vector in cartesian coordinates"""
        return np.array(
            self.left_coord.to("cart") - self.right_coord.to("cart")
        )

    @property
    def dist(self) -> Distance:
        """
        Euclidean distance between the images in ImagePair

        Returns:
            (Distance): Distance in Angstrom
        """
        return Distance(np.linalg.norm(self.dist_vec), units="ang")
