import os

import numpy as np
import pytest

from autode.bracket.imagepair import BaseImagePair
from autode import Molecule
from autode.utils import work_in
from autode import methods
from ..testutils import requires_with_working_xtb_install, work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(here, "data")
# todo replace with zip later


@work_in(datadir)
def test_imgpair_alignment():
    # with same molecule, alignment should produce same coordinates
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_reactant_rotated.xyz")
    imgpair = BaseImagePair(mol1, mol2)

    # alignment happens on init
    new_mol1, new_mol2 = imgpair._left_image, imgpair._right_image
    # left image should have been rotated to align perfectly
    assert np.allclose(new_mol1.coordinates, new_mol2.coordinates, atol=1.0e-5)
    # right image should be translated only, i.e. all difference same
    diff = mol2.coordinates - new_mol2.coordinates
    assert np.isclose(diff, diff[0]).all()
    # now check a random bond distance
    bond_orig = mol1.distance(0, 2)
    bond_new = new_mol1.distance(0, 2)
    assert abs(bond_new - bond_orig) < 0.001


@work_in(datadir)
def test_imgpair_sanity_check():
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_reactant_rotated.xyz")
    mol3 = Molecule(smiles="CCCO")
    mol4 = Molecule("da_reactant_shuffled.xyz")

    # different mol would raise Error
    with pytest.raises(ValueError, match="same number of atoms"):
        _ = BaseImagePair(mol1, mol3)

    # different charge would raise Error
    mol1.charge = -2
    with pytest.raises(ValueError, match="Charge/multiplicity/solvent"):
        _ = BaseImagePair(mol1, mol2)
    mol1.charge = 0

    # different multiplicity would also raise Error
    mol1.mult = 3
    with pytest.raises(ValueError, match="Charge/multiplicity/solvent"):
        _ = BaseImagePair(mol1, mol2)
    mol1.mult = 1

    # different solvents would raise
    mol1.solvent = "water"
    with pytest.raises(ValueError, match="Charge/multiplicity/solvent"):
        _ = BaseImagePair(mol1, mol2)
    mol1.solvent = None

    # different atom order should also raise Error
    with pytest.raises(ValueError, match="order of atoms"):
        _ = BaseImagePair(mol1, mol4)


@requires_with_working_xtb_install
@work_in(datadir)
def test_calc_energy_and_engrad():
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_product.xyz")

    imgpair = BaseImagePair(left_image=mol1, right_image=mol2)
    # without setting method, assert will be set off
    with pytest.raises(AssertionError):
        imgpair.update_one_img_mol_energy("left")

    imgpair.set_method_and_n_cores(engrad_method=methods.XTB(), n_cores=1)
    imgpair.update_one_img_mol_energy("left")
    # only energy should be updated
    assert imgpair.left_coord.e is not None
    assert imgpair.right_coord.e is None
    # units should be forced to Hartree
    assert str(imgpair.left_coord.e.units) == "Unit(Ha)"

    imgpair.update_one_img_mol_engrad("right")
    # energy and gradient of right side would be updated
    assert imgpair.right_coord.e is not None
    assert imgpair.right_coord.g is not None
    assert str(imgpair.right_coord.e.units) == "Unit(Ha)"
    assert str(imgpair._right_image.gradient.units) == "Unit(Ha(Å)^-1)"
    assert imgpair.left_coord.g is None

    # since imgpair takes a copy of initial species they
    # should not be affected
    # todo change if species grad is not changed
    assert mol1.energy is None
    assert mol2.energy is None
    assert mol1.gradient is None
    assert mol2.gradient is None


@requires_with_working_xtb_install
@work_in(datadir)
def test_calc_hessian():
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_product.xyz")

    imgpair = BaseImagePair(left_image=mol1, right_image=mol2)
    imgpair.set_method_and_n_cores(engrad_method=methods.XTB(), n_cores=1)
    # without setting hessian method, assert will be set off
    with pytest.raises(AssertionError):
        imgpair.update_one_img_mol_hess_by_calc("left")

    imgpair.set_method_and_n_cores(
        engrad_method=methods.XTB(), hess_method=methods.XTB(), n_cores=1
    )
    imgpair.update_one_img_mol_hess_by_calc("left")
    # only hessian of left image should be updated
    assert imgpair.left_coord.h is not None
    assert imgpair.left_coord.e is None
    assert imgpair.left_coord.g is None
    # right image should be unchanged
    assert imgpair.right_coord.e is None
    assert imgpair.right_coord.g is None
    assert imgpair.right_coord.h is None


@requires_with_working_xtb_install
def test_hessian_update():
    mol1 = Molecule(smiles="N#N")
    mol2 = Molecule(smiles="N#N")

    imgpair = BaseImagePair(mol1, mol2)
    imgpair.set_method_and_n_cores(
        engrad_method=methods.XTB(), n_cores=1, hess_method=methods.XTB()
    )

    imgpair.update_one_img_mol_engrad("left")
    imgpair.update_one_img_mol_hess_by_calc("left")
    assert imgpair.left_coord.h is not None

    coord = imgpair.left_coord.copy()
    coord[2] += 0.2

    imgpair.left_coord = coord
    with pytest.raises(AssertionError, match="Gradient should"):
        imgpair.update_one_img_mol_hess_by_formula("left")

    assert imgpair.left_coord.h is None
    imgpair.update_one_img_mol_engrad("left")
    imgpair.update_one_img_mol_hess_by_formula("left")
    assert imgpair.left_coord.h is not None
    assert imgpair.right_coord.h is None  # check that it modified current side

    # calling Hessian update again will raise exception
    with pytest.raises(AssertionError):
        imgpair.update_one_img_mol_hess_by_formula("left")
