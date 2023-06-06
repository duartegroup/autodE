"""
Line search optimisers used to solve the 1D optimisation problem by
taking steps :math:`X_{i+1} = X_i + \\alpha p` , where α is a step size and p is
a search direction. See e.g.
https://www.numerical.rl.ac.uk/people/nimg/oumsc/lectures/uepart2.2.pdf
"""
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

from autode.log import logger
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.optimisers.base import Optimiser

if TYPE_CHECKING:
    from autode.opt.coordinates import OptCoordinates
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class LineSearchOptimiser(Optimiser, ABC):
    """Optimiser for a 1D line search in a direction"""

    def __init__(
        self,
        maxiter: int = 10,
        direction: Optional[np.ndarray] = None,
        coords: Optional["OptCoordinates"] = None,
        init_alpha: float = 1.0,
    ):
        """
        Line search optimiser

        -----------------------------------------------------------------------
        Keyword Arguments:
            maxiter (int): Maximum number of iterations to perform

            direction (np.ndarray | None): Direction which to move the
                      coordinates. Shape must be broadcastable to the
                      coordinates. If None then will guess a sensible direction

            coords (autode.opt.coordinates.OptCoordinates | None): Initial
                    coordinates. If None then they will be initialised from the
                    species at runtime

            init_alpha (float): Initial step size
        """
        super().__init__(maxiter=maxiter, coords=coords)

        self.p: Optional[np.ndarray] = direction  # Direction
        self.alpha = init_alpha  # Step size

    @classmethod
    def optimise(
        cls,
        species: "Species",
        method: "Method",
        n_cores: Optional[int] = None,
        coords: Optional["OptCoordinates"] = None,
        direction: Optional[np.ndarray] = None,
        maxiter: int = 5,
        **kwargs: Any,
    ) -> None:
        """
        Optimise a species along a single direction. If the direction is
        unspecified then guess the direction as just the steepest decent
        direction.
        """

        optimiser = cls(maxiter=maxiter, direction=direction, coords=coords)
        optimiser.run(species, method, n_cores=n_cores)

        return None

    @abstractmethod
    def _initialise_coordinates(self) -> None:
        """Initialise the coordinates, if not already specified"""

    def _initialise_run(self) -> None:
        """
        Initialise running the line search. Allows for both the coordinates
        and search direction to be unspecified.
        """
        if self._coords is None:
            self._initialise_coordinates()

        if self.p is None:
            logger.warning(
                "Line search optimiser was initialised without a "
                "search direction. Using steepest decent direction"
            )
            assert self._coords is not None
            if self._coords.g is None:
                self._update_gradient_and_energy()

            assert self._coords.g is not None, "Gradient must be set"
            self.p = -self._coords.g

        return None

    @property
    def minimum_e_coords(
        self,
    ) -> Optional["OptCoordinates"]:
        """Minimum energy coordinates"""
        return None if len(self._history) == 0 else self._history.minimum

    @property
    def _init_coords(
        self,
    ) -> Optional["OptCoordinates"]:
        """
        Initial coordinates from which this line search was initialised from

        -----------------------------------------------------------------------
        Returns:
            (autode.opt.coordinates.OptCoordinates | None):
        """
        return None if len(self._history) == 0 else self._history[0]


class ArmijoLineSearch(LineSearchOptimiser):
    def __init__(
        self,
        maxiter: int = 10,
        direction: Optional[np.ndarray] = None,
        beta: float = 0.1,
        tau: float = 0.5,
        init_alpha: float = 1.0,
        coords: Optional["OptCoordinates"] = None,
    ):
        """
        Backtracking line search by Armijo. Reduces the step size iteratively
        until the convergence condition is satisfied

        [1] L. Armijo. Pacific J. Math. 16, 1966, 1. DOI:10.2140/pjm.1966.16.1.

        ----------------------------------------------------------------------
        Arguments:
            maxiter (int): Maximum number of iteration to perform. Should be
                           small O(1)

        Keyword Arguments:
            direction (np.ndarray): Direction to search along

            beta (float): β parameter in the line search

            tau (float): τ parameter. Multiplicative factor when reducing the
                         step size.

            init_alpha (float): α_0 parameter. Initial value of the step size.
        """
        super().__init__(maxiter, direction=direction, coords=coords)

        self.beta = float(beta)
        self.tau = float(tau)
        self.alpha = float(init_alpha)

    def _step(self) -> None:
        """Take a step in the line search"""
        assert self._coords is not None and self.p is not None

        self.alpha *= self.tau
        self._coords = self._init_coords + self.alpha * self.p

        return None

    def _initialise_coordinates(self) -> None:
        """Initialise the coordinates if they are not defined already.
        Defaults to CartesianCoordinates"""
        assert self._species is not None
        self._coords = CartesianCoordinates(self._species.coordinates)
        return None

    @property
    def _satisfies_wolfe1(self) -> bool:
        """First Wolfe condition:"""
        assert (
            self._coords is not None
            and self._init_coords is not None
            and self.p is not None
        )
        term_2 = self.alpha * self.beta * np.dot(self._init_coords.g, self.p)
        return self._coords.e < self._init_coords.e + term_2

    @property
    def converged(self) -> bool:
        r"""
        Is the line search converged? Defined by the Wolfe condition

        .. math::

            f(x + \alpha^{(l)} p_k) \le f(x) + \alpha^{(l)} \beta g\cdot p

        where α is the step size at the current iteration (denoted by l) and
        β is a variable parameter. The search direction p, gradient are defined
        for the initial point only. See:
        https://en.wikipedia.org/wiki/Wolfe_conditions

        -----------------------------------------------------------------------
        Returns:
            (bool): Line search converged?
        """
        return self._has_coordinates_and_gradient and self._satisfies_wolfe1


class SArmijoLineSearch(ArmijoLineSearch):
    """Speculative Armijo line search"""

    @property
    def converged(self) -> bool:
        """Is the line search converged? For the speculative search to be
        converged requires at least one iteration, there to be a well in the
        search and that the Wolfe condition is satisfied.

        -----------------------------------------------------------------------
        Returns:
            (bool): Line search converged?
        """

        if self.iteration == 0:
            return False

        if not self._history.contains_energy_rise:
            logger.warning(
                f"Line search has not reached a well yet, " f"{self.alpha:.4f}"
            )
            return False

        return self._has_coordinates_and_gradient and self._satisfies_wolfe1

    def _step(self) -> None:
        r"""
        Take a step in the speculative line search. If the energy is monotonic
        decreasing in the search then take steps that of the form

        .. math::

            \alpha = \tau \alpha_\text{init}

        where :math:`\tau > 1`. But as soon as the energy rises then switch to
        :math:`\tau_{k+1} = 1/\tau_{k}`
        """
        assert self._init_coords is not None and self.p is not None

        func = min if self._history.contains_energy_rise else max

        self.tau = func(self.tau, 1 / self.tau)
        self.alpha *= self.tau

        self._coords = self._init_coords + self.alpha * self.p
        return None


class NullLineSearch(LineSearchOptimiser):
    r"""A dummy line search

    .. math::

        \alpha = \alpha_\text{init}
    """

    def _initialise_coordinates(self) -> None:
        """No coordinates to initialise in a null line search"""

    def _initialise_run(self) -> None:
        """Nothing required to initialise a null line search"""

    def _step(self) -> None:
        """No step required in a null line search"""

    @property
    def minimum_e_coords(
        self,
    ) -> Optional["OptCoordinates"]:
        """
        Minimum energy coordinates are defined to be the true step

        -----------------------------------------------------------------------
        Returns:
            (OptCoordinates): Coordinates
        """
        assert self._init_coords is not None and self.p is not None
        return self._init_coords + self.alpha * self.p

    @property
    def converged(self) -> bool:
        """A null line search is converged on the first iteration"""
        return True
