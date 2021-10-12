from autode.solvent import explicit_solvent
from autode.solvent.solvents import get_solvent
from autode.atoms import Atom
import numpy as np
import pytest


def test_get_explicit_solvent():
    # Implement explicit solvent test here..

    with pytest.raises(NotImplementedError):
        _ = get_solvent(solvent_name='water', explicit=True)
