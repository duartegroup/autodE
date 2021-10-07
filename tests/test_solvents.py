from autode.solvent import solvents
from autode.wrappers.ORCA import orca
from autode.exceptions import SolventNotFound
from autode.point_charges import PointCharge
import pytest
import numpy as np


def test_aval_solvents():

    orca_solvents = solvents.get_available_solvent_names(method=orca)
    assert 'water' in orca_solvents


def test_get_solvent():

    water = solvents.get_solvent(solvent_name='water')
    assert water.name == 'water'
    assert water.smiles == 'O'

    with pytest.raises(SolventNotFound):
        _ = solvents.get_solvent(solvent_name='test_solvent')

    assert not water == None
    assert water == solvents.get_solvent(solvent_name='h2o')
