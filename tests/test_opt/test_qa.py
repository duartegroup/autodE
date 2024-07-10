import os
import numpy as np
import pytest
from autode import Molecule, Atom, Config
from autode.methods import XTB
from autode.opt.optimisers.qa import QAOptimiser
from autode.opt.coordinates import DICWithConstraints
from autode.utils import work_in_tmp_dir
from ..testutils import work_in_zipped_dir, requires_working_xtb_install


here = os.path.dirname(os.path.abspath(__file__))
datazip = os.path.join(here, "data", "opt.zip")


@work_in_zipped_dir(datazip)
def test_trm_step():
    mol = Molecule("opt-test.xyz")
    opt = QAOptimiser(maxiter=10, gtol=0.001, etol=0.001, init_trust=0.1)
    opt._species = mol
    opt._build_internal_coordinates()
    assert isinstance(opt._coords, DICWithConstraints)

    grad = np.loadtxt("opt-test_grad.txt")
    hess = np.loadtxt("opt-test_hess.txt")
    opt._coords.update_g_from_cart_g(grad)
    opt._coords.update_h_from_cart_h(hess)

    opt._step()
    step = np.array(opt._history.final) - np.array(opt._history.penultimate)
    step_size = np.linalg.norm(step)
    # TODO: fix the DIC transform bug
    # assert np.isclose(step_size, 0.1)


@work_in_tmp_dir()
@requires_working_xtb_install
def test_trust_update():
    init_trust = 0.05
    water_atoms = [
        Atom("O", -0.0011, 0.3631, -0.0000),
        Atom("H", -0.8250, -0.1819, -0.0000),
        Atom("H", 0.8261, -0.1812, 0.0000),
    ]
    water = Molecule(atoms=water_atoms)
    opt = QAOptimiser(maxiter=10, gtol=1e-3, etol=1e-4, init_trust=init_trust)

    opt._species = water.copy()
    opt._method = XTB()
    opt._n_cores = Config.n_cores
    opt._initialise_run()
    # store last grad
    last_g = opt._coords.g.copy()
    last_h = opt._coords.h.copy()

    opt._step()
    opt._update_gradient_and_energy()
    last_step = np.array(opt._coords) - np.array(opt._history[-2])
    pred_delta_e = float(np.dot(last_g, last_step))
    pred_delta_e += 0.5 * np.linalg.multi_dot((last_step, last_h, last_step))
    # pred_dE should be around -0.002544605 Ha (depends on xTB version)

    def simulate_energy_change_ratio_update_trust(ratio):
        opt.alpha = init_trust
        opt._history.final.e = (
            opt._history.penultimate.e + ratio * pred_delta_e
        )
        opt._update_trust_radius()

    # should not update if trust update turned off
    opt._trust_update = False
    simulate_energy_change_ratio_update_trust(0.2)
    assert np.isclose(opt.alpha, init_trust)

    opt._trust_update = True
    simulate_energy_change_ratio_update_trust(0.2)
    assert np.isclose(opt.alpha, 0.7 * init_trust)

    simulate_energy_change_ratio_update_trust(0.5)
    assert np.isclose(opt.alpha, init_trust)

    simulate_energy_change_ratio_update_trust(1.0)
    assert (np.linalg.norm(last_step) - init_trust) / init_trust < 0.05
    assert np.isclose(opt.alpha, 1.2 * init_trust)

    simulate_energy_change_ratio_update_trust(1.3)
    assert np.isclose(opt.alpha, init_trust)

    simulate_energy_change_ratio_update_trust(1.8)
    assert np.isclose(opt.alpha, 0.7 * init_trust)
