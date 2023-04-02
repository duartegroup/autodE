from typing import Union, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from autode.values import Distance, GradientRMS
from autode.bracket.imagepair import EuclideanImagePair
from autode.methods import get_hmethod
from autode.log import logger
from autode import Config

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class BaseBracketMethod(ABC):
    """
    Base class for all bracketing methods
    """

    def __init__(
        self,
        initial_species: "Species",
        final_species: "Species",
        maxiter: int = 300,
        dist_tol: Union[Distance, float] = Distance(1.0, "ang"),
        gtol: Union[GradientRMS, float] = GradientRMS(1.0e-3, "ha/ang"),
        cineb_at_conv: bool = False,
    ):
        """
        Bracketing methods find transition state by using two images, one
        for the reactant state and another representing the product state.
        These methods move the images continuously until they converge at
        the transition state (TS), i.e. they bracket the TS from both ends.
        It is optionally possible to run a CI-NEB (with only one intervening
        image) from the end-points of a converged bracketing method
        calculation to get much closer to the actual TS.

        Args:
            initial_species: The "reactant" species
            final_species: The "final" species
            maxiter: Maximum number of energy-gradient evaluations
            dist_tol: The distance tolerance at which the method will stop
            gtol: Gradient tolerance for optimisation steps in the method
            cineb_at_conv: Whether to run a CI-NEB with from the final points
        """
        self.imgpair: Optional[
            "EuclideanImagePair"
        ] = None  # must be set by subclass
        self._species: "Species" = initial_species.copy()

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, units="ang")
        self._gtol = GradientRMS(gtol, units="Ha/ang")

        self._should_run_cineb = bool(cineb_at_conv)

    @property
    def ts_guess(self) -> Optional["Species"]:
        """Get the TS guess from image-pair"""
        return self.imgpair.ts_guess

    @property
    def converged(self) -> bool:
        """Whether the bracketing method has converged or not"""

        # NOTE: In bracketing methods, usually the geometry
        # optimisation is done in separate micro-iterations,
        # which means that the gradient tolerance is checked
        # elsewhere, and only distance criteria is checked here

        if self.imgpair.dist <= self._dist_tol:
            return True
        else:
            return False

    @property
    @abstractmethod
    def _method_name(self):
        """Name of the current method"""

    @property
    @abstractmethod
    def _macro_iter(self) -> int:
        """The number of macro-iterations run with this method"""

    @property
    @abstractmethod
    def _micro_iter(self) -> int:
        """Total number of micro-iterations run with this method"""

    @abstractmethod
    def _initialise_run(self) -> None:
        """Initialise the bracketing method run"""

    @abstractmethod
    def calculate(
        self,
        method: "Method",
        n_cores: int,
    ) -> None:
        """
        Run the bracketing method calculation using the method for
        energy/gradient calculation, with n_cores. Does not run
        CI-NEB calculation
        """

    def run(self) -> None:
        """
        Convenience method to run the bracketing method with the
        default high-level method and n_cores from current state
        of the configuration, and then writes trajectories and
        plots the minimum energy path obtained
        """
        method = get_hmethod()
        n_cores = Config.n_cores
        self.calculate(method=method, n_cores=n_cores)
        if self._should_run_cineb:
            self.run_cineb()
        self.write_trajectories()
        self.plot_energies()
        self.imgpair.ts_guess.print_xyz_file(
            f"{self._method_name}_ts_guess.xyz"
        )
        return None

    def write_trajectories(
        self,
        init_trj_filename: Optional[str] = None,
        final_trj_filename: Optional[str] = None,
        total_trj_filename: Optional[str] = None,
    ) -> None:
        """
        Write trajectories as *.xyz files, one for the initial species,
        one for final species, and one for the whole trajectory, including
        any CI-NEB run from the final end points. The default names for
        the trajectories must be set in individual subclasses
        """
        init_trj_filename = (
            init_trj_filename
            if init_trj_filename is not None
            else f"initial_species_{self._method_name}.trj.xyz"
        )
        final_trj_filename = (
            final_trj_filename
            if final_trj_filename is not None
            else f"final_species_{self._method_name}.trj.xyz"
        )
        total_trj_filename = (
            total_trj_filename
            if total_trj_filename is not None
            else f"total_trajectory_{self._method_name}.trj.xyz"
        )
        self.imgpair.write_trajectories(
            init_trj_filename, final_trj_filename, total_trj_filename
        )

        return None

    def plot_energies(
        self,
        filename: Optional[str] = None,
        distance_metric: str = "relative",
    ) -> None:
        """
        Plot the energies of the bracket method run, taking
        into account any CI-NEB interpolation that may have been
        done.

        The distance metric chooses what the x-axis means;
        "relative" means that the points will be plotted in the order
        in which they appear in the total history, and the x-axis
        numbers will represent the relative distances between two
        adjacent points (giving an approximate reaction coordinate).
        "from_start" will calculate the distance of each point from
        the starting reactant structure and use that as the x-axis
        position. If distance metric is set to "index", then the x-axis
        will simply be integer numbers representing each point in order


        Args:
            filename (str): Name of the file
            distance_metric (str): "relative" or "from_start" (or None)
        """
        filename = (
            filename
            if filename is not None
            else f"{self._method_name}_path_energy_plot.pdf"
        )
        self.imgpair.plot_energies(filename, distance_metric)
        return None

    def run_cineb(self) -> None:
        """
        Run CI-NEB from the end-points of a converged bracketing
        calculation. Uses only one intervening image for the
        CI-NEB calculation (which is okay as the bracketing method
        should bring the ends very close to the TS). The result from
        the CI-NEB calculation is stored as coordinates.
        """
        if not self._macro_iter > 0:
            logger.error(
                f"Must run {self._method_name} calculation before"
                f"running the CI-NEB calculation"
            )
            return None

        if not self.converged:
            logger.warning(
                f"{self._method_name} method has not converged, running a"
                " CI-NEB calculation now may not be efficient. Please"
                " check results carefully."
            )
        else:
            logger.info(
                f"{self._method_name} has converged, running CI-NEB"
                f" calculation from the end points"
            )
        self.imgpair.run_cineb_from_end_points()
        return None
