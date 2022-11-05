import pytest
from autode.species import Molecule
from autode.solvent import solvents, get_solvent
from autode.wrappers.ORCA import orca
from autode.exceptions import SolventNotFound


def test_solvent():

    methane = Molecule(smiles="C")
    methane.solvent = "water"
    assert "water" in repr(methane.solvent).lower()

    # Calculation should be able to handle a solvent given as just a string
    assert hasattr(methane.solvent, "mopac")

    # Unknown solvent should raise an exception
    with pytest.raises(SolventNotFound):
        methane.solvent = "XXXX"

    # Default to implicit solvation
    assert methane.solvent.is_implicit


def test_avail_solvents():

    assert "water" in orca.available_implicit_solvents


def test_get_solvent():

    # Solvent must be implicit or explicit
    with pytest.raises(ValueError):
        _ = get_solvent(solvent_name="water", kind="x")

    water = get_solvent(solvent_name="water", kind="implicit")
    assert water.name == "water"
    assert water.smiles == "O"
    assert "h2o" in water.aliases
    assert "water" in water.aliases
    assert water.dielectric is not None

    with pytest.raises(SolventNotFound):
        _ = get_solvent(solvent_name="test_solvent", kind="implicit")

    assert water is not None
    assert water == get_solvent(solvent_name="h2o", kind="implicit")

    # Must define the number of explicit solvent molecules to add
    with pytest.raises(ValueError):
        _ = get_solvent("water", kind="explicit")

    assert get_solvent("h2o", kind="implicit") != get_solvent(
        "h2o", kind="explicit", num=10
    )


def test_solvent_dielectric():

    water = solvents.get_solvent("water", kind="implicit")
    assert abs(water.dielectric - 78) < 1

    assert solvents.ImplicitSolvent("X", "X", aliases=["X"]).dielectric is None
