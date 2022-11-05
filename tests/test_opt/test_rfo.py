import numpy as np
from autode.species.molecule import Molecule
from autode.wrappers.XTB import XTB
from autode.utils import work_in_tmp_dir
from autode.opt.optimisers.rfo import RFOptimiser
from autode.opt.coordinates import CartesianCoordinates
from ..testutils import requires_with_working_xtb_install
from .setup import Method


class TestRFOOptimiser2D(RFOptimiser):
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
        **kwargs,
    ):
        super().__init__(maxiter=maxiter, etol=etol, gtol=gtol, **kwargs)

        init_arr = np.array([init_x, init_y])
        self._coords = CartesianCoordinates(init_arr)
        self._coords.h = np.eye(2)

        self.e_func = e_func
        self.g_func = g_func

    def _space_has_degrees_of_freedom(self) -> bool:
        return True

    def _log_convergence(self) -> None:
        x, y = self._coords
        print(f"{x:.4f}, {y:.4f}", f"E = {round(self._coords.e, 5)}")

    def _update_gradient_and_energy(self) -> None:

        x, y = self._coords
        self._coords.e = self.e_func(x, y)
        self._coords.g = self.g_func(x, y)

    def _initialise_run(self) -> None:
        self._update_gradient_and_energy()


def test_simple_quadratic_opt():

    optimiser = TestRFOOptimiser2D(
        e_func=lambda x, y: x**2 + y**2,
        g_func=lambda x, y: np.array([2.0 * x, 2.0 * y]),
        init_y=1.0,
        init_x=1.0,
        init_alpha=0.5,
    )
    optimiser.run(Molecule(name="blank"), method=Method())
    assert optimiser.converged
    assert optimiser.iteration < 10


def test_branin_opt():
    def energy(x, y):
        return (y - 0.129 * x**2 + 1.6 * x - 6) ** 2 + 6.07 * np.cos(x) + 10

    def grad(x, y):
        de_dx = 2 * (1.6 - 0.258 * x) * (
            y - 0.129 * x**2 + 1.6 * x - 6
        ) - 6.07 * np.sin(x)

        de_dy = 2 * (y - 0.129 * x**2 + 1.6 * x - 6)

        return np.array([de_dx, de_dy])

    optimiser = TestRFOOptimiser2D(
        e_func=energy, g_func=grad, init_y=14.0, init_x=6.0, init_alpha=2.0
    )
    optimiser.run(Molecule(name="blank"), method=Method())

    assert optimiser.converged
    assert np.allclose(optimiser._coords, np.array([3.138, 2.252]), atol=0.02)

    assert optimiser.iteration < 30


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
@requires_with_working_xtb_install
def test_molecular_opt():

    mol = Molecule(smiles="O")
    assert [atom.label for atom in mol.atoms] == ["O", "H", "H"]

    RFOptimiser.optimise(mol, method=XTB())

    # Check optimised distances are similar to running the optimiser in XTB
    for oh_atom_idx_pair in [(0, 1), (0, 2)]:
        assert np.isclose(
            mol.distance(*oh_atom_idx_pair).to("Å"), 0.9595, atol=1e-2
        )

    assert np.isclose(mol.distance(1, 2), 1.5438, atol=1e-2)
