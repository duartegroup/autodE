from autode.log.methods import methods
from autode.methods import ORCA
from autode import Molecule
from autode.config import Config
from .testutils import work_in_zipped_dir
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_init():

    # Should contain some mention of autodE by default
    assert "autodE" in str(methods)


@work_in_zipped_dir(os.path.join(here, "data", "comp_methods.zip"))
def test_dft():

    orca = ORCA()

    methods.clear()
    h2 = Molecule(smiles="[H][H]", solvent_name="water")
    h2.single_point(method=orca)

    assert "PBE0" in str(methods)
    assert "def2-TZVP" in str(methods)
    assert "4.2.1" in str(methods)

    # Default CPCM solvation in orca
    assert "CPCM" in str(methods)
