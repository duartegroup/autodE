from abc import ABC
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.optimisers.base import NDOptimiser


class SteepestDescent(NDOptimiser, ABC):
    def __init__(self, maxiter, gtol, etol, step_size=0.2, **kwargs):
        """
        Steepest decent optimiser

        ----------------------------------------------------------------------
        Arguments:
            step_size (float): Size of the step to take. Units of distance

        See Also:

            :py:meth:`NDOptimiser <autode.opt.optimisers.base.NDOptimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)

        self.alpha = step_size

    def _step(self) -> None:
        r"""
        Take a steepest decent step:

        .. math::

            x_{i+1} = x_{i} - \alpha \nabla E

        where :math:`\alpha` is the step size.
        """
        assert self._coords is not None, "A step requires set coordinates"
        assert self._coords.g is not None, "A step requires a defined gradient"

        self._coords = self._coords - self.alpha * self._coords.g


class CartesianSDOptimiser(SteepestDescent):
    """Steepest decent optimisation in Cartesian coordinates"""

    def _initialise_run(self) -> None:
        """
        Initialise a set of cartesian coordinates. As a species' coordinates
        are already Cartesian there is nothing special to do
        """
        assert self._species is not None
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._update_gradient_and_energy()


class DIC_SD_Optimiser(SteepestDescent):
    """Steepest decent optimisation in delocalised internal coordinates"""

    def _initialise_run(self) -> None:
        """Initialise the delocalised internal coordinates"""
        assert self._species is not None

        self._coords = CartesianCoordinates(self._species.coordinates).to(
            "dic"
        )
        self._update_gradient_and_energy()
