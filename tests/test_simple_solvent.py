from autode import Molecule
from autode.calculation import get_solvent_name
from autode.exceptions import SolventNotFound
from autode.wrappers.MOPAC import mopac
import pytest


def test_solvent():

    methane = Molecule(smiles='C')
    methane.solvent = 'water'

    # Calculation should be able to handle a solvent given as just a string
    assert get_solvent_name(molecule=methane, method=mopac) == 'water'

    # Unknown solvent should raise an exception
    with pytest.raises(SolventNotFound):
        methane.solvent = 'XXXX'
        get_solvent_name(molecule=methane, method=mopac)
