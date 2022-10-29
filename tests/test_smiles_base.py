import pytest
import autode.exceptions as ex
from autode.smiles.base import (
    SMILESAtom,
    SMILESBond,
    SMILESStereoChem,
    RingBond,
)


def test_smiles_atom():

    atom = SMILESAtom("C", stereochem=SMILESStereoChem.TET_NORMAL)
    atom.invert_stereochem()
    assert atom.stereochem == SMILESStereoChem.TET_INVERTED

    # Invert back
    atom.invert_stereochem()
    assert atom.stereochem == SMILESStereoChem.TET_NORMAL


def test_smiles_bond():

    atoms = [
        SMILESAtom("C", stereochem=SMILESStereoChem.ALKENE_UP),
        SMILESAtom("C", stereochem=SMILESStereoChem.ALKENE_UP),
    ]

    bond = SMILESBond(0, 1, symbol="=")
    assert bond.is_cis(atoms=atoms)
    assert not bond.is_trans(atoms=atoms)

    for atom in atoms:
        atom.stereochem = None

    # Without stereochemistry the double bond should default to trans
    assert not bond.is_cis(atoms=atoms)
    assert bond.is_trans(atoms=atoms)

    # Invalid bond symbol
    with pytest.raises(ex.InvalidSmilesString):
        _ = SMILESBond(0, 1, symbol="--")

    bond = RingBond(0, symbol="=")
    bond.close(1, symbol="-")

    assert bond.closes_ring
    assert bond.in_ring(rings_idxs=[{0, 1}])
