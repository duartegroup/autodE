import os
import numpy as np
import pytest

from autode import Molecule, Atom
from autode.geom import calc_rmsd
from autode.opt.coordinates import CartesianCoordinates
from autode.methods import XTB
from autode.values import Energy
from autode.utils import work_in_tmp_dir
from autode.bracket.imagepair import (
    EuclideanImagePair,
    _calculate_engrad_for_species,
    _calculate_hessian_for_species,
)
from ..testutils import work_in_zipped_dir, requires_working_xtb_install

here = os.path.dirname(os.path.abspath(__file__))
datazip = os.path.join(here, "data", "geometries.zip")


class NullImagePair(EuclideanImagePair):
    """Used for testing"""

    @property
    def ts_guess(self):
        return None


def test_imagpair_coordinates():
    mol = Molecule(smiles="CCO")
    imgpair = NullImagePair(mol, mol.copy())

    # error on setting wrong type of coordinates, even if
    # it's an array of right shape
    coord_array = np.array(mol.coordinates.flatten())
    with pytest.raises(TypeError):
        imgpair.left_coords = coord_array
    with pytest.raises(TypeError):
        imgpair.right_coords = coord_array

    coords = CartesianCoordinates(mol.coordinates)
    coords += 0.1
    # no error if Cartesian coordinates
    imgpair.left_coords = coords
    coords = CartesianCoordinates(np.arange(mol.n_atoms + 1))
    num = mol.n_atoms * 3
    with pytest.raises(ValueError, match=f"Must have {num} entries"):
        imgpair.left_coords = coords
    with pytest.raises(ValueError, match=f"Must have {num} entries"):
        imgpair.right_coords = coords


def test_imagepair_method_typing():
    mol = Molecule(smiles="CCO")
    imgpair = NullImagePair(mol, mol.copy())

    with pytest.raises(TypeError, match="method needs to be of type"):
        imgpair.set_method_and_n_cores(method=5, n_cores=2)

    with pytest.raises(TypeError, match="hessian method needs to be of type"):
        imgpair.set_method_and_n_cores(method=XTB(), n_cores=2, hess_method=5)


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
def test_imgpair_energy_plotting(caplog):
    mol1 = Molecule(smiles="CCO")
    mol2 = Molecule(smiles="CCO")

    imgpair = NullImagePair(mol1, mol2)
    imgpair.left_coords.e = Energy(-3.14)
    imgpair.right_coords.e = Energy(-2.87)
    # spoof new coordinates
    imgpair.left_coords = imgpair.left_coords * 0.99
    imgpair.right_coords = imgpair.right_coords * 0.99
    imgpair.left_coords.e = Energy(-1.99)
    imgpair.right_coords.e = Energy(-2.15)
    # if CINEB run at the end
    imgpair._cineb_coords = imgpair.left_coords * 0.99
    imgpair._cineb_coords.e = Energy(-0.99)

    # test all distance_metrics
    imgpair.plot_energies(filename="test0.pdf", distance_metric="relative")
    assert os.path.isfile("test0.pdf")
    imgpair.plot_energies(filename="test1.pdf", distance_metric="from_start")
    assert os.path.isfile("test1.pdf")
    imgpair.plot_energies(filename="test2.pdf", distance_metric="index")
    assert os.path.isfile("test2.pdf")

    # distance metric should be one of the three options
    with pytest.raises(KeyError):
        imgpair.plot_energies(filename="test.pdf", distance_metric="abc")

    # if any energy is missing, also no plotting should be done
    imgpair.left_coords.e = None
    with caplog.at_level("ERROR"):
        imgpair.plot_energies(
            filename="test_noE.pdf", distance_metric="relative"
        )
    assert not os.path.isfile("test-noE.pdf")
    assert "do not have associated energies" in caplog.text


@work_in_tmp_dir()
def test_imgpair_trajectory_plotting():
    mol1 = Molecule(smiles="CCO")
    mol2 = Molecule(smiles="CCO")

    imgpair = NullImagePair(mol1, mol2)
    imgpair.left_coords = imgpair.left_coords * 0.99
    imgpair.right_coords = imgpair.right_coords * 0.99
    imgpair._cineb_coords = imgpair.left_coords * 0.99

    imgpair.print_geometries("init.xyz", "fin.xyz", "total.xyz")
    assert os.path.isfile("init.xyz")
    assert os.path.isfile("fin.xyz")
    assert os.path.isfile("total.xyz")


@requires_working_xtb_install
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
    assert imgpair.left_coords.e is not None
    assert imgpair.right_coords.e is not None
    # units should be forced to Hartree
    assert str(imgpair.left_coords.e.units) == "Unit(Ha)"
    assert str(imgpair.right_coords.e.units) == "Unit(Ha)"
    # gradient should also be updated
    assert imgpair.left_coords.g is not None
    assert imgpair.right_coords.g is not None

    # since imgpair takes a copy of initial species they
    # should not be affected
    assert mol1.energy is None
    assert mol2.energy is None
    assert mol1.gradient is None
    assert mol2.gradient is None


@requires_working_xtb_install
@work_in_tmp_dir()
def test_imgpair_calc_hess():
    mol1 = Molecule(smiles="N#N")

    imgpair = NullImagePair(mol1, mol1.copy())
    imgpair.set_method_and_n_cores(method=XTB(), n_cores=1, hess_method=XTB())
    imgpair.update_both_img_hessian_by_calc()

    # should not change gradient and energy
    assert imgpair.left_coords.g is None
    assert imgpair.left_coords.e is None
    # only hessian should be calculation
    assert imgpair.left_coords.h is not None


@requires_working_xtb_install
def test_calculation_functions():
    # Test the external functions that are used in image pair for
    # easy parallelisation
    mol = Molecule(smiles="CCO")
    en, grad = _calculate_engrad_for_species(mol, XTB(), 1)
    assert en == mol.energy
    assert grad == mol.gradient

    mol = Molecule(smiles="N#N")
    hess = _calculate_hessian_for_species(mol, XTB(), 1)
    # should take a copy to calculate
    assert mol.energy is None
    assert mol.gradient is None
    assert mol.hessian is None
    assert hess is not None
    assert hess.shape == (3 * mol.n_atoms, 3 * mol.n_atoms)


@work_in_zipped_dir(datazip)
def test_hessian_update():
    mol1 = Molecule(
        atoms=[
            Atom("N", 0.5588, 0.0000, 0.0000),
            Atom("N", -0.5588, 0.0000, 0.0000),
        ]
    )
    g = np.loadtxt("n2_grad.txt")
    h = np.loadtxt("n2_hess.txt")

    imgpair = NullImagePair(mol1, mol1.copy())
    imgpair.set_method_and_n_cores(method=XTB(), n_cores=1, hess_method=XTB())

    imgpair.left_coords.g = g.copy()
    imgpair.left_coords.h = h.copy()
    imgpair.right_coords.g = g.copy()
    imgpair.right_coords.h = h.copy()

    assert imgpair.left_coords.h is not None

    coord = imgpair.left_coords.copy()
    coord[0] += 0.1

    imgpair.left_coords = coord
    imgpair.right_coords = coord

    new_g = np.loadtxt("n2_new_grad.txt")

    assert imgpair.left_coords.h is None
    imgpair.left_coords.g = new_g.copy()
    imgpair.right_coords.g = new_g.copy()

    # update the hessian with update formula
    imgpair.update_both_img_hessian_by_formula()
    assert imgpair.left_coords.h is not None
    assert imgpair.right_coords.h is not None
