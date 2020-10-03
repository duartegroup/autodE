from autode.log.methods import methods
from autode.methods import ORCA
from autode import Molecule
from .testutils import work_in_zipped_dir


def test_init():

    # Should contain some mention of autodE by default
    assert 'autodE' in str(methods)
    assert 'autodE' in methods.string()


# @work_in_zipped_dir()
def test_dft():

    methods.clear()
    h2 = Molecule(smiles='[H][H]', solvent_name='water')
    h2.single_point(method=ORCA())

    assert 'PBE0' in str(methods)
    assert 'def2-TZVP' in str(methods)
    assert '4.2.1' in str(methods)

    # Default CPCM solvation in orca
    assert 'CPCM' in str(methods)

    print(methods)
