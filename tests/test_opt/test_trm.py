import os
import numpy as np
import pytest
from autode.species import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.values import Energy
from autode.utils import work_in_tmp_dir
from ..testutils import requires_with_working_xtb_install, work_in_zipped_dir
from autode.opt.coordinates import CartesianCoordinates, DIC
from autode.opt.optimisers.trm import (
    HybridTRMOptimiser,
    CartesianHybridTRMOptimiser,
)

here = os.path.dirname(os.path.abspath(__file__))


@work_in_zipped_dir(os.path.join(here, "data", "opt.zip"))
def test_trm_step1():
    mol = Molecule("opt-test.xyz")
    opt = HybridTRMOptimiser(maxiter=10, gtol=0.001, etol=0.001)
    opt._species = mol
    opt._build_internal_coordinates()
    assert isinstance(opt._coords, DIC)

    grad = np.loadtxt("opt-test_grad.txt")
    hess = np.loadtxt("opt-test_hess.txt")
    opt._coords.update_g_from_cart_g(grad)
    opt._coords.update_h_from_cart_h(hess)

    opt._step()
    step = opt._history.final.to("cart") - opt._history.penultimate.to("cart")
    step_size = np.linalg.norm(step)
    assert np.isclose(step_size, opt.alpha, rtol=0.01)  # 1% error margin

    opt = CartesianHybridTRMOptimiser(maxiter=10, gtol=0.001, etol=0.001)
    opt._species = mol
    opt._coords = CartesianCoordinates(mol.coordinates)

    opt._coords.update_g_from_cart_g(grad)
    opt._coords.update_h_from_cart_h(hess)

    opt._step()
    assert isinstance(opt._coords, CartesianCoordinates)
    step = opt._history.final.to("cart") - opt._history.penultimate.to("cart")
    step_size = np.linalg.norm(step)
    assert np.isclose(step_size, opt.alpha, rtol=0.001)  # 0.1% error margin


def test_damping_in_hybridtrm_optimiser():
    coord1 = CartesianCoordinates([1.0, -2.0, 1.0, 3.0, 1.1, 1.2])
    coord1.e = 0.10
    coord1.g = np.array([0.2, 0.3, 0.1, 0.2, 0.3, 0.4])
    coord1.h = np.eye(6)
    coord2 = CartesianCoordinates([1.1, -1.9, 1.1, 3.1, 1.2, 1.3])
    coord2.e = 0.14
    coord2.g = np.array([0.1, 0.2, 0.01, 0.1, 0.2, 0.3])
    coord2.h = np.eye(6)
    opt = CartesianHybridTRMOptimiser(maxiter=2, gtol=0.1, etol=0.1)

    # simulate an oscillation happening
    opt._coords = coord1
    opt._coords = coord2
    opt._coords = coord1
    opt._coords = coord2
    opt._coords = coord1
    opt._coords = coord2

    assert opt._is_oscillating()  # should register as oscillation
    opt._step()  # should take damped step
    # 50% mixing
    avg_coord = (coord1 + coord2) / 2.0
    assert np.allclose(avg_coord, opt._coords)


def test_trim_convergence_gtol_overachieved():
    # if rms grad is <= gtol/10, assumed to be converged, even if
    # delta E criteria is not met
    opt = CartesianHybridTRMOptimiser(maxiter=3, gtol=0.1, etol=0.01)
    coord1 = CartesianCoordinates([1.0, -2.0, 1.0, 3.0, 1.1, 1.2])
    coord1.e = Energy(0.10, "Ha")
    coord1.g = np.array([0.01, 0.03, 0.01, 0.02, 0.03, 0.04])
    coord2 = coord1.copy()
    coord2.e = Energy(0.05, "Ha")
    coord2.g = coord1.g / 10
    opt._coords = coord1
    opt._coords = coord2

    # energy criteria not achieved
    assert opt._abs_delta_e > opt.etol
    # grad criteria overachieved
    assert opt._g_norm <= opt.gtol / 10
    assert opt.converged


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_trm_molecular_opt():
    mol = Molecule(smiles="O")
    assert [atom.label for atom in mol.atoms] == ["O", "H", "H"]

    HybridTRMOptimiser.optimise(mol, method=XTB(), gtol=1.0e-4)

    # Check optimised distances are similar to running the optimiser in XTB
    for oh_atom_idx_pair in [(0, 1), (0, 2)]:
        assert np.isclose(
            mol.distance(*oh_atom_idx_pair).to("Å"), 0.9595, atol=1e-3
        )

    assert np.isclose(mol.distance(1, 2), 1.5438, atol=1e-3)

    mol = Molecule(smiles="O")
    CartesianHybridTRMOptimiser.optimise(mol, method=XTB(), gtol=1e-4)

    # Same check for Cartesian
    for oh_atom_idx_pair in [(0, 1), (0, 2)]:
        assert np.isclose(
            mol.distance(*oh_atom_idx_pair).to("Å"), 0.9595, atol=1e-3
        )

    assert np.isclose(mol.distance(1, 2), 1.5438, atol=1e-3)


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_trust_update(caplog):
    init_trust = 0.05
    water_atoms = [
        Atom("O", -0.0011, 0.3631, -0.0000),
        Atom("H", -0.8250, -0.1819, -0.0000),
        Atom("H", 0.8261, -0.1812, 0.0000),
    ]
    water = Molecule(atoms=water_atoms)
    opt = HybridTRMOptimiser(
        maxiter=10, gtol=1.0e-3, etol=1.0e-4, init_trust=init_trust
    )
    opt._species = water.copy()  # take copy so original is not modified
    opt._method = XTB()
    opt._n_cores = 1
    opt._initialise_run()
    # store last grad
    last_g = opt._coords.g.reshape(-1, 1).copy()
    last_h = opt._coords.h.copy()

    opt._step()
    opt._update_gradient_and_energy()
    last_step = opt._history[-1].raw - opt._history[-2].raw
    last_cart_step = opt._history[-1].to("cart") - opt._history[-2].to("cart")
    cart_step_size = np.linalg.norm(last_cart_step)
    pred_delta_e = float(last_g.T @ last_step)
    pred_delta_e += 0.5 * (last_step.T @ last_h @ last_step)
    # pred_dE should be around -0.0025446056 Ha (may change with different version of xTB)

    def simulate_energy_change_ratio_update_trust(ratio):
        opt.alpha = init_trust
        opt._history.final.e = (
            opt._history.penultimate.e + ratio * pred_delta_e
        )
        opt._update_trust_radius()

    # if update is not allowed, there shouldn't be update
    opt._upd_alpha = False
    simulate_energy_change_ratio_update_trust(0.2)
    assert np.isclose(opt.alpha, init_trust)
    opt._upd_alpha = True

    simulate_energy_change_ratio_update_trust(0.2)
    assert np.isclose(opt.alpha, 0.5 * min(init_trust, cart_step_size))

    simulate_energy_change_ratio_update_trust(0.5)
    assert np.isclose(opt.alpha, init_trust)

    simulate_energy_change_ratio_update_trust(1.0)
    assert abs(cart_step_size - init_trust) / init_trust < 0.05
    assert np.isclose(opt.alpha, 1.414 * init_trust)

    simulate_energy_change_ratio_update_trust(1.3)
    assert np.isclose(opt.alpha, init_trust)

    simulate_energy_change_ratio_update_trust(1.7)
    assert np.isclose(opt.alpha, 0.5 * min(init_trust, cart_step_size))

    # if energy change too high > 2.0 or too low < -1.0, trust radius
    # is decreased, and last step is rejected
    with caplog.at_level("WARNING"):
        simulate_energy_change_ratio_update_trust(2.2)
    assert "rejecting last geometry step" in caplog.text
    assert np.isclose(opt.alpha, 0.5 * min(init_trust, cart_step_size))
    # should remove last coords, bringing back original state
    assert opt.iteration == 0
    assert np.allclose(
        opt._history.final.to("cart"), water.coordinates.flatten()
    )


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_optimiser_plotting():
    mol = Molecule(smiles="O")

    opt = HybridTRMOptimiser(maxiter=100, gtol=1e-3, etol=1e-3)
    opt.run(mol, method=XTB())

    opt.plot_optimisation()
    assert os.path.isfile(f"{mol.name}_opt_plot.pdf")


@work_in_tmp_dir()
def test_optimiser_xyz_trajectory():
    mol = Molecule(smiles="N#N")
    opt = CartesianHybridTRMOptimiser(maxiter=10, gtol=1e-3, etol=1e-3)
    coord1 = CartesianCoordinates(mol.coordinates)
    opt._coords = coord1

    with pytest.raises(AssertionError):
        # no species should raise error
        opt.print_geometries()

    opt._species = mol
    assert opt.iteration == 0
    # xyz trajectory only if more than 1 coordinates present
    opt.print_geometries()
    assert not os.path.isfile(f"{mol.name}_opt.trj.xyz")

    coord2 = coord1.copy() * 0.9
    opt._coords = coord2
    assert opt.iteration == 1
    opt.print_geometries()
    assert os.path.isfile(f"{mol.name}_opt.trj.xyz")


@work_in_tmp_dir()
def test_optimiser_plotting_sanity_checks(caplog):
    mol = Molecule(smiles="N#N")

    opt = CartesianHybridTRMOptimiser(maxiter=10, gtol=1e-3, etol=1e-3)
    coord1 = CartesianCoordinates(mol.coordinates)
    coord1.e = Energy(0.1, "Ha")
    coord1.update_g_from_cart_g(np.array([0.01, 0.02, 0.05, 0.06, 0.03, 0.07]))
    opt._coords = coord1
    opt._species = mol
    assert opt.iteration == 0
    assert not opt.converged
    # plotting does not work if less than 2 points
    with caplog.at_level("WARNING"):
        opt.plot_optimisation(filename="test_plot0.pdf")
    assert not os.path.isfile("test_plot0.pdf")
    assert "Less than 2 points, cannot draw optimisation" in caplog.text

    opt._coords = coord1.copy()
    # either rms_grad or energy plot has to be requested
    with caplog.at_level("WARNING"):
        opt.plot_optimisation("test_plot1.pdf", False, False)
    assert not os.path.isfile("test_plot1.pdf")
    assert not opt.converged
    assert "Must plot either energies or RMS gradients" in caplog.text
    assert "Optimisation is not converged, drawing a plot" in caplog.text
