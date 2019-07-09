from autode import config
from autode import ORCAio
import os
here = os.path.dirname(os.path.abspath(__file__))


def test_xtb_install():
    os.chdir(os.path.join(here, 'orca'))

    assert os.path.exists(config.Config.path_to_orca)

    orca_test_out_lines = ORCAio.run_orca(inp_filename='h.inp', out_filename='h.out')
    assert len(orca_test_out_lines) > 0
    assert 'ORCA TERMINATED NORMALLY' in orca_test_out_lines[-2]
