import numpy as np
import pytest

from autode.bracket.imagepair import BaseImagePair
from autode import Molecule
from autode.utils import work_in
from ..testutils import requires_with_working_xtb_install


@work_in("data")
def test_imgpair_alignment():
    # with same molecule, alignment should produce same coordinates
    mol1 = Molecule('da_reactant.xyz')
    mol2 = Molecule('da_reactant_rotated.xyz')
    imgpair = BaseImagePair(mol1, mol2)

    # alignment happens on init
    new_mol1, new_mol2 = imgpair._left_image, imgpair._right_image
    # left image should have been rotated to align perfectly
    assert np.allclose(new_mol1.coordinates, new_mol2.coordinates, atol=1.e-5)
    # right image should be translated only, i.e. all difference same
    diff = mol2.coordinates - new_mol2.coordinates
    assert np.isclose(diff, diff[0]).all()
    # now check a random bond distance
    bond_orig = mol1.distance(0, 2)
    bond_new = new_mol1.distance(0, 2)
    assert abs(bond_new - bond_orig) < 0.001


@work_in("data")
def test_imgpair_sanity_check():
    mol1 = Molecule('da_reactant.xyz')
    mol2 = Molecule('da_reactant_rotated.xyz')
    mol3 = Molecule(smiles='CCCO')
    mol4 = Molecule('da_reactant_shuffled.xyz')

    # different mol would raise Error
    with pytest.raises(ValueError):
        _ = BaseImagePair(mol1, mol3)

    # different charge would raise Error
    mol1.charge = -2
    with pytest.raises(ValueError):
        _ = BaseImagePair(mol1, mol2)
    mol1.charge = 0

    # different multiplicity would also raise Error
    mol1.mult = 3
    with pytest.raises(ValueError):
        _ = BaseImagePair(mol1, mol2)
    mol1.mult = 1

    # different atom order should also raise Error
    with pytest.raises(ValueError):
        _ = BaseImagePair(mol1, mol4)


@requires_with_working_xtb_install
def test_set_energy_gradient():
    pass