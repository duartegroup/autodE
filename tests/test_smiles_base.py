import pytest
import autode.exceptions as ex
from autode.smiles.base import SMILESAtom, SMILESBond


def test_smiles_atom():

    atom = SMILESAtom('C', stereochem='@')
    atom.invert_stereochem()
    assert atom.stereochem == '@@'

    # Invert back
    atom.invert_stereochem()
    assert atom.stereochem == '@'


def test_smiles_bond():

    atoms = [SMILESAtom('C', stereochem='al_up'),
             SMILESAtom('C', stereochem='al_up')]

    bond = SMILESBond(0, 1, symbol='=')
    assert bond.is_cis(atoms=atoms)
    assert not bond.is_trans(atoms=atoms)

    for atom in atoms:
        atom.stereochem = None

    # Without stereochemistry the double bond should default to trans
    assert not bond.is_cis(atoms=atoms)
    assert bond.is_trans(atoms=atoms)

    # Invalid bond symbol
    with pytest.raises(ex.InvalidSmilesString):
        _ = SMILESBond(0, 1, symbol='--')
