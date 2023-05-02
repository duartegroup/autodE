import os
import numpy as np
import pytest

from autode import Molecule
from autode.geom import calc_rmsd
from autode.methods import XTB
from autode.values import Energy
from autode.utils import work_in, work_in_tmp_dir
from autode.bracket.imagepair import EuclideanImagePair
from ..testutils import work_in_zipped_dir, requires_with_working_xtb_install

here = os.path.dirname(os.path.abspath(__file__))
datazip = os.path.join(here, "data", "geometries.zip")


class NullImagePair(EuclideanImagePair):
    """Used for testing"""

    @property
    def ts_guess(self):
        return None


@work_in_zipped_dir(datazip)
def test_imgpair_alignment():
    # with same molecule, alignment should produce same coordinates
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_reactant_rotated.xyz")
    imgpair = NullImagePair(mol1, mol2)

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


@work_in_zipped_dir(datazip)
def test_imgpair_sanity_check():
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_reactant_rotated.xyz")
    mol3 = Molecule(smiles="CCCO")
    mol4 = Molecule("da_reactant_shuffled.xyz")

    # different mol would raise Error
    with pytest.raises(ValueError, match="same number of atoms"):
        _ = NullImagePair(mol1, mol3)

    # different charge would raise Error
    mol1.charge = -2
    with pytest.raises(ValueError, match="Charge/multiplicity/solvent"):
        _ = NullImagePair(mol1, mol2)
    mol1.charge = 0

    # different multiplicity would also raise Error
    mol1.mult = 3
    with pytest.raises(ValueError, match="Charge/multiplicity/solvent"):
        _ = NullImagePair(mol1, mol2)
    mol1.mult = 1

    # different solvents would raise
    mol1.solvent = "water"
    with pytest.raises(ValueError, match="Charge/multiplicity/solvent"):
        _ = NullImagePair(mol1, mol2)
    mol1.solvent = None

    # different atom order should also raise Error
    with pytest.raises(ValueError, match="order of atoms"):
        _ = NullImagePair(mol1, mol4)


@work_in_zipped_dir(datazip)
def test_imgpair_distance():
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_product.xyz")
    imgpair = NullImagePair(mol1, mol2)
    rmsd = calc_rmsd(mol1.coordinates, mol2.coordinates)
    dist = rmsd * np.sqrt(mol1.n_atoms * 3)
    assert np.isclose(dist, imgpair.dist, rtol=1e-8)


@work_in_tmp_dir()
def test_energy_plotting_and_trajectory_ignored_if_less_than_three_points():
    mol1 = Molecule(smiles="CCO")
    mol2 = Molecule(smiles="CCO")
    imgpair = NullImagePair(mol1, mol2)

    # file should not be written with only two points
    imgpair.plot_energies(filename="test.pdf", distance_metric="relative")
    assert not os.path.isfile("test.pdf")

    imgpair.print_geometries("init.xyz", "fin.xyz", "total.xyz")
    assert not os.path.isfile("init.xyz")
    assert not os.path.isfile("fin.xyz")
    assert not os.path.isfile("total.xyz")


@work_in_tmp_dir()
def test_imgpair_energy_plotting():
    mol1 = Molecule(smiles="CCO")
    mol2 = Molecule(smiles="CCO")

    imgpair = NullImagePair(mol1, mol2)
    imgpair.left_coord.e = Energy(-3.14)
    imgpair.right_coord.e = Energy(-2.87)
    # spoof new coordinates
    imgpair.left_coord = imgpair.left_coord * 0.99
    imgpair.right_coord = imgpair.right_coord * 0.99
    imgpair.left_coord.e = Energy(-1.99)
    imgpair.right_coord.e = Energy(-2.15)
    # if CINEB run at the end
    imgpair._cineb_coords = imgpair.left_coord * 0.99
    imgpair._cineb_coords.e = Energy(-0.99)

    # test all distance_metrics
    imgpair.plot_energies(filename="test0.pdf", distance_metric="relative")
    assert os.path.isfile("test0.pdf")
    imgpair.plot_energies(filename="test1.pdf", distance_metric="from_start")
    assert os.path.isfile("test1.pdf")
    imgpair.plot_energies(filename="test2.pdf", distance_metric="index")
    assert os.path.isfile("test2.pdf")


@work_in_tmp_dir()
def test_imgpair_trajectory_plotting():
    mol1 = Molecule(smiles="CCO")
    mol2 = Molecule(smiles="CCO")

    imgpair = NullImagePair(mol1, mol2)
    imgpair.left_coord = imgpair.left_coord * 0.99
    imgpair.right_coord = imgpair.right_coord * 0.99
    imgpair._cineb_coords = imgpair.left_coord * 0.99

    imgpair.print_geometries("init.xyz", "fin.xyz", "total.xyz")
    assert os.path.isfile("init.xyz")
    assert os.path.isfile("fin.xyz")
    assert os.path.isfile("total.xyz")


@requires_with_working_xtb_install
@work_in_zipped_dir(datazip)
def test_imgpair_calc_engrad():
    mol1 = Molecule("da_reactant.xyz")
    mol2 = Molecule("da_product.xyz")

    imgpair = NullImagePair(left_image=mol1, right_image=mol2)
    # without setting method, assert will be set off
    with pytest.raises(AssertionError):
        imgpair.update_both_img_engrad()

    imgpair.set_method_and_n_cores(method=XTB(), n_cores=1)
    imgpair.update_both_img_engrad()
    # energy should be updated
    assert imgpair.left_coord.e is not None
    assert imgpair.right_coord.e is not None
    # units should be forced to Hartree
    assert str(imgpair.left_coord.e.units) == "Unit(Ha)"
    assert str(imgpair.right_coord.e.units) == "Unit(Ha)"
    # gradient should also be updated
    assert imgpair.left_coord.g is not None
    assert imgpair.right_coord.g is not None

    # since imgpair takes a copy of initial species they
    # should not be affected
    assert mol1.energy is None
    assert mol2.energy is None
    assert mol1.gradient is None
    assert mol2.gradient is None


@requires_with_working_xtb_install
def test_hessian_update():
    # todo clean up with stored hessian in txt instead of calling hessian
    # todo separate hessian calculation into another test
    mol1 = Molecule(smiles="N#N")
    mol2 = Molecule(smiles="N#N")

    imgpair = NullImagePair(mol1, mol2)
    imgpair.set_method_and_n_cores(method=XTB(), n_cores=1, hess_method=XTB())

    imgpair.update_both_img_engrad()
    imgpair.update_both_img_hessian_by_calc()
    assert imgpair.left_coord.h is not None

    coord = imgpair.left_coord.copy()
    coord[2] += 0.2

    imgpair.left_coord = coord
    imgpair.right_coord = coord

    assert imgpair.left_coord.h is None
    imgpair.update_both_img_engrad()
    imgpair.update_both_img_mol_hessian_by_formula()
    assert imgpair.left_coord.h is not None
    assert (
        imgpair.right_coord.h is not None
    )  # check that it modified current side

    # calling Hessian update again will raise exception
    with pytest.raises(AssertionError):
        imgpair.update_both_img_mol_hessian_by_formula()
