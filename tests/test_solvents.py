import pytest
from autode.species import Molecule
from autode.solvent import solvents
from autode.wrappers.ORCA import orca
from autode.exceptions import SolventNotFound


def test_solvent():

    methane = Molecule(smiles='C')
    methane.solvent = 'water'

    # Calculation should be able to handle a solvent given as just a string
    assert hasattr(methane.solvent, 'mopac')

    # Unknown solvent should raise an exception
    with pytest.raises(SolventNotFound):
        methane.solvent = 'XXXX'


def test_avail_solvents():

    assert 'water' in orca.available_implicit_solvents


def test_get_solvent():

    water = solvents.get_solvent(solvent_name='water')
    assert water.name == 'water'
    assert water.smiles == 'O'

    with pytest.raises(SolventNotFound):
        _ = solvents.get_solvent(solvent_name='test_solvent')

    assert not water is None
    assert water == solvents.get_solvent(solvent_name='h2o')


def test_solvent_dielectric():

    water = solvents.get_solvent('water', implicit=True)
    assert abs(water.dielectric - 78) < 1

    assert solvents.ImplicitSolvent('X', 'X', aliases=['X']).dielectric is None
