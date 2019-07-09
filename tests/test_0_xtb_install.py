from autode import config
from autode import XTBio
import os
here = os.path.dirname(os.path.abspath(__file__))


def test_xtb_install():
    os.chdir(os.path.join(here, 'xtb'))

    assert os.path.exists(config.Config.path_to_xtb)

    xtb_test_out_lines = XTBio.run_xtb('h2.xyz')
    assert len(xtb_test_out_lines) > 0
