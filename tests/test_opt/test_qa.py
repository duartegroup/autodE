import os
import numpy as np
import pytest
from autode import Molecule
from autode.opt.optimisers.qa import QAOptimiser
from autode.opt.coordinates import DICWithConstraints
from ..testutils import work_in_zipped_dir


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
    assert np.isclose(step_size, 0.1)
