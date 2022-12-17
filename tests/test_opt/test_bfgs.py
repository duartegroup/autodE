"""
https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
"""
import numpy as np
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.optimisers.bfgs import BFGSOptimiser
from autode.opt.optimisers.line_search import (
    NullLineSearch,
    ArmijoLineSearch,
    SArmijoLineSearch,
)
from autode.species import Molecule
from .setup import Method


class TestBFGSOptimiser2D(BFGSOptimiser):
    """Simple 2D optimiser using a BFGS update step"""

    __test__ = False

    def __init__(
        self,
        e_func,
        g_func,
        init_x,
        init_y,
        maxiter=30,
        etol=1e-4,
        gtol=1e-3,
        line_search_type=NullLineSearch,
        init_alpha=1.0,
    ):
        super().__init__(
            maxiter=maxiter,
            line_search_type=line_search_type,
            init_alpha=init_alpha,
            etol=etol,
            gtol=gtol,
        )

        init_arr = np.array([init_x, init_y])
        self._coords = CartesianCoordinates(init_arr)

        self.e_func = e_func
        self.g_func = g_func

    def _space_has_degrees_of_freedom(self) -> bool:
        return True

    def _log_convergence(self) -> None:
        x, y = self._coords
        print(
            f"{x:.4f}, {y:.4f}",
            f"E = {round(self._coords.e, 5)},"
            # f' {np.round(self._coords.g, 5)}'
        )

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.e = self.e_func(x, y)
        self._coords.g = self.g_func(x, y)

    def _initialise_run(self) -> None:

        # Guess the Hessian as the identity matrix
        self._update_gradient_and_energy()
        self._coords.h = np.eye(len(self._coords))

    @property
    def converged(self) -> bool:
        return np.linalg.norm(self._coords.g) < self._gtol


class TestBFGSOptimiser(TestBFGSOptimiser2D):
    def __init__(self, **kwargs):
        super().__init__(
            e_func=lambda x, y: x**2 + y**2,
            g_func=lambda x, y: np.array([2.0 * x, 2.0 * y]),
            init_y=1.0,
            init_x=1.0,
            **kwargs,
        )


def test_simple_quadratic_opt():

    optimiser = TestBFGSOptimiser()
    method = Method()
    assert method.is_available

    optimiser.run(Molecule(name="blank"), method=Method())
    assert optimiser.converged


def test_inv_hessian_update():

    # Initial imperfect guess of the Hessian matrix for E = x^2 + y^2
    init_h = np.array([[1.0, 0.0], [0.0, 1.0]])

    optimiser = TestBFGSOptimiser()
    optimiser._species, optimiser._method = Molecule(name="blank"), Method()
    assert optimiser.iteration == 0

    optimiser._update_gradient_and_energy()
    optimiser._coords.h = init_h.copy()

    optimiser._step()

    # Should take the inverse of the guess after the first step
    assert np.allclose(optimiser._history[-1].h_inv, np.linalg.inv(init_h))

    # Then for the new set of coordinates generate a better guess of the
    # inverse Hessian once the gradient has been updated
    optimiser._update_gradient_and_energy()

    h_inv_true = np.array([[0.5, 0.0], [0.0, 0.5]])

    assert np.linalg.norm(
        optimiser._updated_h_inv() - h_inv_true
    ) < np.linalg.norm(np.linalg.inv(init_h) - h_inv_true)

    optimiser.run(Molecule(name="blank"), method=Method())
    assert optimiser.converged
    assert np.isclose(optimiser._coords.e, 0.0, atol=1e-5)

    # By the end of the optimisation the Hessian should be pretty good
    assert np.allclose(optimiser._coords.h_inv, h_inv_true, atol=1)


def test_quadratic_opt():
    """E = x^2 + y^2 + xy/10,  ∇E  = (2x + 0.1y, 2y + 0.1x)"""

    optimiser = TestBFGSOptimiser2D(
        e_func=lambda x, y: x**2 + y**2 + x * y / 10.0,
        g_func=lambda x, y: np.array([2.0 * x + 0.1 * y, 2.0 * y + 0.1 * x]),
        init_x=1.0,
        init_y=1.0,
    )
    optimiser.run(Molecule(name="blank"), method=Method())
    assert optimiser.converged
    assert optimiser._coords.e < 1e-3
    assert np.allclose(
        optimiser._coords, np.zeros(2), atol=1e-3  # Minimum is at (0, 0)
    )


def check_gaussian_well_opt(init_x, init_y):
    """E = -exp(-(x^2 + y^2)), ∇E  = (2xE, 2yE)"""

    def energy(x, y):
        return -np.exp(-(x**2 + y**2))

    def grad(x, y):
        e = np.exp(-(x**2 + y**2))
        return np.array([2.0 * x * e, 2.0 * y * e])

    def hessian(x, y):
        h_xy = -4.0 * x * y * np.exp(-(x**2) - y**2)
        return np.array(
            [
                [2 * (1 - 2.0 * x**2) * np.exp(-(x**2) - y**2), h_xy],
                [h_xy, 2 * (1 - 2.0 * y**2) * np.exp(-(x**2) - y**2)],
            ]
        )

    class LineSearch(SArmijoLineSearch):
        def _update_gradient_and_energy(self) -> None:
            self._coords.e = energy(*self._coords)
            self._coords.g = grad(*self._coords)

    optimiser = TestBFGSOptimiser2D(
        e_func=energy,
        g_func=grad,
        init_x=init_x,
        init_y=init_y,
        line_search_type=LineSearch,
    )

    optimiser.run(Molecule(name="blank"), method=Method())

    assert optimiser.converged
    assert np.isclose(optimiser._coords.e, -1)  # Minimum is -1
    assert np.allclose(
        optimiser._coords, np.zeros(2), atol=1e-3  # Minimum is at (0, 0)
    )

    # Hessian should be close-ish to the true
    assert np.allclose(
        optimiser._coords.h, hessian(*optimiser._coords), atol=1
    )


def test_gaussian_well_opt():

    for (x, y) in [(-0.1, 0.0), (-1.0, 0.0), (2.0, 1.0), (2.0, -2.0)]:
        check_gaussian_well_opt(init_x=x, init_y=y)


def test_complex_2d_opt():
    def energy(x, y):
        return 10 * (y - x**2) ** 2 + (x - 1) ** 2

    def grad(x, y):
        return np.array(
            [2 * (20 * x**3 - 20 * x * y + x - 1), 20 * (y - x**2)]
        )

    optimiser = TestBFGSOptimiser2D(
        e_func=energy,
        g_func=grad,
        init_x=-1.0,
        init_y=1.0,
        init_alpha=0.1,
        maxiter=100,
    )
    optimiser.run(Molecule(name="blank"), method=Method())

    assert optimiser.converged
    assert np.allclose(
        optimiser._coords,
        np.array([1.0, 1.0]),
        atol=0.1,  # Minimum is at (1, 1)
    )
