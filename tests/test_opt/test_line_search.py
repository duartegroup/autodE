import shutil
import numpy as np
from autode.atoms import Atom
from autode.species import Molecule
from autode.utils import work_in_tmp_dir
from autode.methods import XTB
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.optimisers.line_search import (
    ArmijoLineSearch,
    NullLineSearch,
    LineSearchOptimiser,
)
from ..testutils import requires_with_working_xtb_install
from .setup import Method


def quadratic(x, y):
    return x**2 + y**2, np.array([2 * x, 2 * y])


class TestSDLineSearch(LineSearchOptimiser):
    """Line search for E = x^2 + y^2"""

    __test__ = False

    def __init__(
        self,
        init_alpha=1.0,
        energy_grad_func=quadratic,
        direction=None,
        coords=None,
    ):
        super().__init__(
            maxiter=10,
            direction=direction,
            init_alpha=init_alpha,
            coords=coords,
        )

        self.energy_grad_func = energy_grad_func

    @property
    def converged(self) -> bool:
        """Simple convergence criteria"""
        return (
            self.iteration > 0
            and self._coords.e is not None
            and self._coords.e < 0.01
        )

    def _log_convergence(self) -> None:
        pass  # print(self._coords.e)

    def _initialise_coordinates(self) -> None:
        self._coords = CartesianCoordinates(np.array([1.1, 0.2]))

    def _step(self) -> None:
        self._coords = self._coords + self.alpha * self.p

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.e, self._coords.g = self.energy_grad_func(x, y)


class TestArmijoLineSearch(ArmijoLineSearch):

    __test__ = False

    def __init__(self, init_step_size=1.0, energy_grad_func=quadratic):
        super().__init__(maxiter=10, init_alpha=init_step_size)

        self.energy_grad_func = energy_grad_func

    def _initialise_coordinates(self) -> None:
        self._coords = CartesianCoordinates(np.array([-0.8, 1.0]))

    def _update_gradient_and_energy(self) -> None:
        return TestSDLineSearch._update_gradient_and_energy(self)

    def _log_convergence(self) -> None:
        pass  # print(self._e_prev, self._species.energy)


def test_null_line_search():

    ls = NullLineSearch(init_alpha=0.1)
    assert ls.converged

    # Dummy (empty) coordinates should be initialisable without a species
    assert ls._initialise_coordinates() is None

    ls.run(Molecule(name="blank"), Method())
    assert np.isclose(ls.alpha, 0.1)


def test_simple_line_search():

    blank_mol = Molecule(name="blank")
    blank_method = Method()

    optimiser = TestSDLineSearch(init_alpha=0.1)
    optimiser.run(blank_mol, method=blank_method)
    assert optimiser.converged

    # Minimum is at (0, 0). Should be close to that
    assert np.allclose(optimiser._coords, np.array([0.0, 0.0]), atol=1e-1)


def test_armijo_line_search_default():

    optimiser = TestArmijoLineSearch()
    assert not optimiser.converged

    optimiser.run(Molecule(name="blank"), method=Method())
    assert optimiser.converged

    # Minimum is at (0, 0). Should be close to that point with 0 energy
    assert np.allclose(optimiser._coords, np.array([0.0, 0.0]))
    assert np.isclose(optimiser._coords.e, 0.0)


def test_armijo_line_search_diff_step_sizes():

    # using different step sizes should also converge
    for init_step_size in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0):
        optimiser = TestArmijoLineSearch(init_step_size=init_step_size)
        optimiser.run(Molecule(name="blank"), method=Method())

        assert optimiser.converged


def test_armijo_line_search_complex_func():
    def energy_grad(x, y):
        energy = 10 * (y - x**2) ** 2 + (x - 1) ** 2
        gradient = np.array(
            [2 * (20 * x**3 - 20 * x * y + x - 1), 20 * (y - x**2)]
        )

        return energy, gradient

    optimiser = TestArmijoLineSearch(
        energy_grad_func=energy_grad, init_step_size=0.1
    )
    optimiser.run(Molecule(name="blank"), method=Method())

    assert optimiser.converged
    assert optimiser._coords.e < optimiser._init_coords.e


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_xtb_h2_cart_opt():

    optimiser = ArmijoLineSearch()
    assert not optimiser.converged

    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=1.5)])

    # Should not converge in only two steps
    ArmijoLineSearch.optimise(species=h2, method=XTB(), maxiter=2)
    assert not optimiser.converged

    # Line search should step in the direction to reduce the distance
    assert h2.distance(0, 1) < 1.4
