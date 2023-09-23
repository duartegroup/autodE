from typing import Union, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from autode.values import Distance, GradientRMS
from autode.bracket.imagepair import EuclideanImagePair
from autode.log import logger
from autode.utils import work_in
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
            final_species: The "product" species
            maxiter: Maximum number of energy-gradient evaluations
            dist_tol: The distance tolerance at which the method
                      will stop, in units of Å if not given
            gtol: Gradient tolerance for optimisation steps in
                  the method, units Ha/Å if not given
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
    def _name(self) -> str:
        """Name of the current bracketing method, obtained from class name"""
        return type(self).__name__

    @property
    def ts_guess(self) -> Optional["Species"]:
        """Get the TS guess from image-pair"""
        assert self.imgpair is not None, "Must have an image pair for TS guess"
        return self.imgpair.ts_guess

    @property
    def converged(self) -> bool:
        """Whether the bracketing method has converged or not"""
        assert self.imgpair is not None, "Must have an image pair"

        # NOTE: Usually geometry optimisation is done in separate
        # micro-iters, so gradient is checked elsewhere
        return self.imgpair.dist <= self._dist_tol

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
        assert self.imgpair is not None, "Must have an image pair to log"

        logger.info(
            f"{self._name} Macro-iteration #{self._macro_iter}: "
            f"Distance = {self.imgpair.dist:.4f}; Energy (initial species) = "
            f"{self.imgpair.left_coords.e:.6f}; Energy (final species) = "
            f"{self.imgpair.right_coords.e:.6f}"
        )

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        """Whether it has exceeded the number of maximum micro-iterations"""
        if self._micro_iter >= self._maxiter:
            logger.error(
                f"Reached the maximum number of micro-iterations "
                f"*{self._maxiter}"
            )
            return True
        else:
            return False

    def calculate(
        self,
        method: "Method",
        n_cores: Optional[int] = None,
    ) -> None:
        """
        Run the bracketing method calculation using the method for
        energy/gradient calculation, with n_cores. Runs CI-NEB at
        the end if requested; then save the .xyz trajectories,
        plot the energies and finally save the peak as TS guess.
        This function should be called only once!

        Args:
            method (Method): Method used for calculating energy/gradients
            n_cores (int): Number of cores to use for calculation
        """

        @work_in(self._name.lower())
        def run():
            self._calculate(method, n_cores)

        run()
        return None

    def _calculate(
        self,
        method: "Method",
        n_cores: Optional[int] = None,
    ) -> None:
        """
        Actually runs the calculation, it is wrapped around in calculate()
        so that the results are placed in a sub-folder
        """
        assert self.imgpair is not None, "Must have set image pair"

        n_cores = Config.n_cores if n_cores is None else int(n_cores)
        self.imgpair.set_method_and_n_cores(method, n_cores)
        self._initialise_run()

        logger.info(f"Starting {self._name} method to find transition state")

        while not self.converged:
            self._step()

            if self.imgpair.has_jumped_over_barrier:
                # TODO: implement image pair regeneration
                logger.error(
                    "One image has probably jumped over the barrier, in"
                    f" {self._name} TS search. Please check the"
                    f" results carefully"
                )
                if self._barrier_check:
                    logger.info(f"Stopping {self._name} calculation")
                    break

            if self._exceeded_maximum_iteration:
                break

            self._log_convergence()

        # exited main loop, run CI-NEB if required and bracket converged
        if self._should_run_cineb:
            if self.converged and not self.imgpair.has_jumped_over_barrier:
                self.run_cineb()
            else:
                logger.warning(
                    f"{self._name} calculation has not converged"
                    f" properly or one side has jumped over the barrier,"
                    f" skipping CI-NEB run"
                )

        logger.info(
            f"Finished {self._name} procedure in {self._macro_iter} "
            f"macro-iterations consisting of {self._micro_iter} micro-"
            f"iterations (optimiser steps). {self._name} is "
            f"{'converged' if self.converged else 'not converged'}"
        )

        self.print_geometries()
        self.plot_energies()
        if self.converged and self.ts_guess is not None:
            self.ts_guess.print_xyz_file(filename=f"{self._name}_ts_guess.xyz")
        return None

    def print_geometries(
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
        assert self.imgpair is not None, "Must have an image pair to plot"

        init_trj_filename = (
            init_trj_filename
            if init_trj_filename is not None
            else f"initial_species_{self._name}.trj.xyz"
        )
        final_trj_filename = (
            final_trj_filename
            if final_trj_filename is not None
            else f"final_species_{self._name}.trj.xyz"
        )
        total_trj_filename = (
            total_trj_filename
            if total_trj_filename is not None
            else f"total_trajectory_{self._name}.trj.xyz"
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
            filename (str|None): Name of the file (optional)
            distance_metric (str): "relative" or "from_start" or "index"
        """
        assert self.imgpair is not None, "Must have an image pair to plot"

        filename = (
            filename
            if filename is not None
            else f"{self._name}_path_energy_plot.pdf"
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
        assert self.imgpair is not None, "Must have image pair to run CINEB"

        if not self._micro_iter > 0:
            logger.error(
                f"Must run {self._name} calculation before"
                f"running the CI-NEB calculation"
            )
            return None

        if not self.converged or self.imgpair.dist > 2.0:
            logger.warning(
                f"{self._name} method has not converged sufficiently,"
                f" running a CI-NEB calculation now may cause errors."
                f" Please check results carefully."
            )
        else:
            logger.info(
                f"{self._name} has converged, running CI-NEB"
                f" calculation from the end points"
            )
        self.imgpair.run_cineb_from_end_points()
        return None
