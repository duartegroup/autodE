from typing import Union, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from autode.values import Distance, GradientRMS
from autode.neb import CINEB
from autode.bracket.imagepair import ImagePair
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
        self.imgpair = ImagePair(initial_species, final_species)
        self._species: "Species" = initial_species.copy()

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, units="ang")
        self._gtol = GradientRMS(gtol, units="Ha/ang")

        self._end_cineb = bool(cineb_at_conv)
        self._ci_coords = None  # to hold the CI-NEB coordinates

    @property
    def converged(self) -> bool:
        if self.imgpair.dist <= self._dist_tol:
            return True
        else:
            return False

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
        energy/gradient calculation, with n_cores.
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
        self.write_trajectories()
        self.plot_energies()

    def write_trajectories(self) -> None:
        """
        Write trajectories as *.xyz files, one for the initial species,
        one for final species, and one for the whole trajectory, including
        any CI-NEB run from the final end points
        """
        # todo write in imagepair, and then make imagepair private
        pass

    def plot_energies(self) -> None:
        pass

    def run_cineb(self) -> None:
        """
        Run CI-NEB from the end-points of a converged bracketing
        calculation. Uses only one intervening image for the
        CI-NEB calculation (which is okay as the bracketing method
        should bring the ends very close to the TS). The result from
        the CI-NEB calculation is stored as coordinates.
        """
        # todo put all in imagepair, and then remove func,check in calculate()
        if self._ci_coords is not None:
            logger.warning("CI-NEB calculation was already run before")

        if not self.converged:
            logger.warning(
                "The bracketing method has not converged, please check"
                "the results of CI-NEB carefully"
            )

        self._ci_coords = self.imgpair.run_cineb_and_get_final_coords()
