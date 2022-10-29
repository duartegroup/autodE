import os
import numpy as np
from autode.input_output import xyz_file_to_atoms
from autode.transition_states.ts_guess import TSguess
from autode.methods import ORCA
from . import testutils


here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(
    os.path.join(here, "data", "constrained_opt.zip")
)
@testutils.requires_with_working_xtb_install
def test_constrained_opt():

    ts_guess = TSguess(atoms=xyz_file_to_atoms("h_shift_init.xyz"), mult=2)
    ts_guess.run_constrained_opt(
        name="const_opt",
        distance_consts={(2, 6): 1.304, (6, 1): 1.295},
        method=ORCA(),
    )

    assert np.isclose(ts_guess.distance(2, 6), 1.304, atol=0.1)
    assert np.isclose(ts_guess.distance(6, 1), 1.295, atol=0.1)
