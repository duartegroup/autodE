import numpy as np
from autode.species.molecule import Molecule
from autode.wrappers.base import Method
from autode.opt.optimisers.rfo import RFOOptimiser
from autode.opt.coordinates import CartesianCoordinates


class TestRFOOptimiser2D(RFOOptimiser):
    """Simple 2D optimiser using a BFGS update step"""

    __test__ = False

    def __init__(self, e_func, g_func, init_x, init_y,
                 maxiter=30, etol=1E-4, gtol=1E-3, **kwargs):
        super().__init__(maxiter=maxiter, etol=etol, gtol=gtol, **kwargs)

        init_arr = np.array([init_x, init_y])
        self._coords = CartesianCoordinates(init_arr)
        self._coords.h = np.eye(2)

        self.e_func = e_func
        self.g_func = g_func

    def _log_convergence(self) -> None:
        x, y = self._coords
        print(f'{x:.4f}, {y:.4f}', f'E = {round(self._coords.e, 5)}')

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.e = self.e_func(x, y)
        self._coords.g = self.g_func(x, y)

    def _initialise_run(self) -> None:
        self._update_gradient_and_energy()


def test_simple_quadratic_opt():

    optimiser = TestRFOOptimiser2D(e_func=lambda x, y: x**2 + y**2,
                                   g_func=lambda x, y: np.array([2.0*x, 2.0*y]),
                                   init_y=1.0, init_x=1.0,
                                   init_alpha=0.5)
    optimiser.run(Molecule(name='blank'), method=Method())
    assert optimiser.converged
    assert optimiser.iteration < 10


def test_branin_opt():

    def energy(x, y):
        return (y - 0.129 * x ** 2 + 1.6 * x - 6) ** 2 + 6.07 * np.cos(x) + 10

    def grad(x, y):
        de_dx = (2*(1.6 - 0.258 * x)*(y - 0.129 * x ** 2 + 1.6 * x - 6)
                 - 6.07 * np.sin(x))

        de_dy = 2 * (y - 0.129 * x ** 2 + 1.6 * x - 6)

        return np.array([de_dx, de_dy])

    optimiser = TestRFOOptimiser2D(e_func=energy,
                                   g_func=grad,
                                   init_y=14.0, init_x=6.0,
                                   init_alpha=2.0)
    optimiser.run(Molecule(name='blank'), method=Method())

    assert optimiser.converged
    assert np.allclose(optimiser._coords,
                       np.array([3.138, 2.252]),
                       atol=0.02)

    assert optimiser.iteration < 30
