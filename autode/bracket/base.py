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
        barrier_check: bool = True,
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
            barrier_check: Whether to stop the calculation if one image is
                           detected to have jumped over the barrier. Do not
                           turn this off unless you are absolutely sure!
        """
        # imgpair type must be set by subclass
        self.imgpair: Optional["EuclideanImagePair"] = None
        self._species: "Species" = initial_species.copy()

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, units="ang")
        self._gtol = GradientRMS(gtol, units="Ha/ang")

        self._should_run_cineb = bool(cineb_at_conv)
        self._barrier_check = bool(barrier_check)

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
    def _step(self) -> None:
        """
        One step of the bracket method, with one macro-iteration
        and multiple micro-iterations. This must also set new
        coordinates for the next step
        """

    def _log_convergence(self) -> None:
        """
        Log the convergence of the bracket method. Only logs macro-iters,
        subclasses may implement further logging for micro-iters
        """
        logger.info(
            f"Macro-iteration #{self._macro_iter}: Distance = "
            f"{self.imgpair.dist:.4f}; Energy (initial species) = "
            f"{self.imgpair.left_coord.e:.6f}; Energy (final species) = "
            f"{self.imgpair.right_coord.e:.6f}"
        )

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        """Whether it has exceeded the number of maximum micro-iterations"""
        if self._micro_iter >= self._maxiter:
            return True
        else:
            return False

    @abstractmethod
    def calculate(
        self,
        method: "Method",
        n_cores: int,
    ) -> None:
        """
        Run the bracketing method calculation using the method for
        energy/gradient calculation, with n_cores. Runs CI-NEB at
        the end if requested
        """
        self.imgpair.set_method_and_n_cores(method, n_cores)
        self._initialise_run()

        logger.info(
            f"Starting {self._method_name} method to find transition state"
        )

        while not self.converged:
            self._step()

            if self.imgpair.has_jumped_over_barrier():
                logger.warning(
                    "One image has probably jumped over the barrier, in"
                    f" {self._method_name} TS search. Please check the"
                    f" results carefully"
                )
                if self._barrier_check:
                    break

            if self._exceeded_maximum_iteration:
                break

            self._log_convergence()

        # exited main loop, run CI-NEB if required and bracket converged
        if self._should_run_cineb and self.converged:
            self.run_cineb()

        # print final message
        logger.info(
            f"Finished {self._method_name} procedure in {self._macro_iter} "
            f"macro-iterations consisting of {self._micro_iter} micro-"
            f"iterations (optimiser steps). {self._method_name} is "
            f"{'converged' if self.converged else 'not converged'}"
        )
        return None

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
        self.imgpair.print_geometries(
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
