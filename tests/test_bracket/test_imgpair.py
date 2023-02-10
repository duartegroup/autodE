import numpy as np

from autode.bracket.imagepair import BaseImagePair
from autode import Molecule
from autode.utils import work_in


@work_in("data")
def test_imgpair_alignment():
    # with same molecule, alignment should produce same coordinates
    # also distances should stay the same
    mol1 = Molecule('sn2_rct.xyz', charge=-1)
    mol2 = Molecule('sn2_rct_rotated.xyz', charge=-1)
    imgpair = BaseImagePair(mol1, mol2)

    # alignment happens on init
    new_mol1, new_mol2 = imgpair._left_image, imgpair._right_image
    # left image should have been rotated to align perfectly
    assert np.allclose(new_mol1.coordinates, new_mol2.coordinates, atol=1.e-6)
    # right image should be translated only, i.e. all difference same
    diff = mol2.coordinates - new_mol2.coordinates
    assert np.isclose(diff, diff[0]).all()
    # now check bond distances

