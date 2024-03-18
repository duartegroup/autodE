import copy
import os
import zipfile

import pytest
import numpy as np

from autode.methods import XTB
from autode.calculations.types import CalculationType
from autode.values import GradientRMS, PotentialEnergy
from autode.species.molecule import Molecule
from autode.hessians import Hessian
from autode.utils import work_in_tmp_dir
from ..testutils import requires_working_xtb_install
from .molecules import h2, methane_mol, h_atom
from .setup import Method
from autode.utils import NumericStringDict
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.base import OptimiserHistory, NullOptimiser
from autode.opt.optimisers.steepest_descent import (
    CartesianSDOptimiser,
    DIC_SD_Optimiser,
)


def sample_cartesian_optimiser():
    return CartesianSDOptimiser(
        maxiter=1, gtol=GradientRMS(0.1), etol=PotentialEnergy(0.1)
    )


def test_optimiser_construct():
    # Optimiser needs a Species
    with pytest.raises(ValueError):
        sample_cartesian_optimiser().run(species=None, method=XTB())

    # also a method
    with pytest.raises(ValueError):
        sample_cartesian_optimiser().run(species=methane_mol(), method=None)

    # Optimiser needs valid arguments
    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(
            maxiter=0, gtol=GradientRMS(0.1), etol=PotentialEnergy(0.1)
        )

    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(
            maxiter=1, gtol=GradientRMS(-0.1), etol=PotentialEnergy(0.1)
        )

    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(
            maxiter=1, gtol=GradientRMS(-0.1), etol=PotentialEnergy(0.1)
        )

    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(
            maxiter=1, gtol=GradientRMS(0.1), etol=PotentialEnergy(-0.1)
        )


def test_initialise_species_and_method():
    optimiser = sample_cartesian_optimiser()

    # Species and method need to be valid
    with pytest.raises(ValueError):
        optimiser._initialise_species_and_method(species=None, method=None)

    with pytest.raises(ValueError):
        optimiser._initialise_species_and_method(species="a", method=None)


def test_coords_set():
    optimiser = sample_cartesian_optimiser()

    # Internal set of coordinates must be an instance of OptCoordinate
    with pytest.raises(ValueError):
        optimiser._coords = "a"


def test_abs_diff_e():
    # Define a intermediate optimiser state with two sets of coordinates
    optimiser = sample_cartesian_optimiser()
    optimiser._history.add(CartesianCoordinates([0.0, 1.0]))
    optimiser._history.add(CartesianCoordinates([0.0, 1.1]))

    # 2nd iteration for a history of two, indexed from 0
    assert optimiser.iteration == 1

    # without defined energies |E_0 - E_1| cannot be calculated
    assert optimiser._abs_delta_e

    # but can be if both structures have a potential energy
    optimiser._history.final.e = PotentialEnergy(-1.0)
    optimiser._history.penultimate.e = PotentialEnergy(-1.1)

    diff_e = optimiser._abs_delta_e
    assert isinstance(diff_e, PotentialEnergy)

    assert np.isclose(diff_e, 0.1, atol=1e-6)


def test_g_norm():
    optimiser = sample_cartesian_optimiser()

    # With no coordinates the norm of the gradient is infinity
    assert optimiser._coords is None
    assert not np.isfinite(optimiser._g_norm)

    # Likewise if the gradient is unset
    optimiser._coords = CartesianCoordinates([1.0, 0.0, 0.0])
    assert optimiser._coords.g is None
    assert not np.isfinite(optimiser._g_norm)


def test_optimiser_h_update():
    optimiser = sample_cartesian_optimiser()

    # Remove any possible updater type
    optimiser._hessian_update_types = []

    c1 = CartesianCoordinates([1.0, 0.0, 0.0])
    c1.h = np.eye(3)

    optimiser._history.add(c1)

    c2 = CartesianCoordinates([1.1, 0.0, 0.0])
    c2.h = np.eye(3)

    optimiser._history.add(c2)

    # and try and update the (inverse) hessian, which is impossible without
    # an updater
    with pytest.raises(Exception):
        _ = optimiser._updated_h_inv()


def test_history():
    optimiser = sample_cartesian_optimiser()
    assert optimiser.iteration < 1
    assert len(optimiser._history) < 1

    # Cannot get the final set of coordinates without any history
    with pytest.raises(IndexError):
        _ = optimiser._history.final

    # or the ones before that
    with pytest.raises(IndexError):
        _ = optimiser._history.penultimate


@work_in_tmp_dir()
@requires_working_xtb_install
def test_xtb_h2_cart_opt():
    mol = h2()
    CartesianSDOptimiser.optimise(mol, method=XTB(), maxiter=50)

    # Optimised H-H distance is ~0.7 Å
    assert np.isclose(mol.distance(0, 1), 0.777, atol=0.1)


@work_in_tmp_dir()
@requires_working_xtb_install
def test_xtb_h2_cart_opt():
    optimiser = CartesianSDOptimiser(
        maxiter=2,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(1e-3),
    )
    optimiser._species = h2()
    optimiser._coords = CartesianCoordinates(optimiser._species.coordinates)

    assert not optimiser.converged

    # Should not converge in only two steps
    optimiser.run(method=XTB(), species=h2())
    assert not optimiser.converged


@work_in_tmp_dir()
@requires_working_xtb_install
def test_xtb_h2_dic_opt():
    # In DICs we can use a much larger step size
    optimiser = DIC_SD_Optimiser(
        step_size=2.5,
        maxiter=10,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(0.0001),
    )

    mol = h2()
    # Should optimise fast, in only a few steps
    optimiser.run(species=mol, method=XTB())

    assert optimiser.converged
    assert optimiser.iteration < 10
    assert np.isclose(mol.distance(0, 1), 0.77, atol=0.1)


class HarmonicPotentialOptimiser(CartesianSDOptimiser):
    def _update_gradient_and_energy(self):
        self._species.coordinates = self._coords.to("cart")
        r = self._species.distance(0, 1)
        self._coords.e = self._species.energy = (r - 2.0) ** 2
        self._coords.g = np.array([-0.01, 0.0, 0.0, 0.01, 0.0, 0.0])


@work_in_tmp_dir()
def test_callback_function():
    mol = h2()

    def func(coords, m=None):
        m.print_xyz_file(filename="tmp.xyz")
        assert os.path.exists("tmp.xyz")

    optimiser = HarmonicPotentialOptimiser(
        maxiter=1,
        callback=func,
        callback_kwargs={"m": mol},
        gtol=GradientRMS(0.1),
        etol=PotentialEnergy(0.1),
    )

    optimiser.run(species=mol, method=Method())


@work_in_tmp_dir()
def test_last_energy_change_with_no_steps():
    mol = h2()
    optimiser = HarmonicPotentialOptimiser(
        maxiter=2, gtol=GradientRMS(999), etol=PotentialEnergy(999)
    )

    optimiser.run(mol, method=Method())
    assert optimiser.converged
    assert optimiser.last_energy_change < 1


def test_value_extraction_from_string():
    value = 99.9
    s = f"E = {value}"  # " =" is implied
    assert np.isclose(NumericStringDict(s)["E"], value)


@work_in_tmp_dir()
@requires_working_xtb_install
def test_optimisation_is_possible_with_single_atom():
    mol = h_atom()
    CartesianSDOptimiser.optimise(mol, method=XTB(), maxiter=2)
    assert mol.energy is None


class ConvergedHarmonicPotentialOptimiser(CartesianSDOptimiser):
    @property
    def converged(self) -> bool:
        return True


class UnconvergedHarmonicPotentialOptimiser(CartesianSDOptimiser):
    @property
    def converged(self) -> bool:
        return False


def test_last_energy_change_less_than_two_steps():
    optimiser = ConvergedHarmonicPotentialOptimiser(
        maxiter=2, gtol=GradientRMS(999), etol=PotentialEnergy(999)
    )

    coords = CartesianCoordinates(np.zeros(1))
    coords.e = 0
    coords.g = np.zeros_like(coords)
    optimiser._coords = coords

    assert optimiser.converged
    assert np.isclose(optimiser.last_energy_change, 0.0)

    optimiser.__class__ = UnconvergedHarmonicPotentialOptimiser
    assert not optimiser.converged
    assert not np.isfinite(optimiser.last_energy_change)


class HessianInTesting(Hessian):
    """Hessian with a different class, used for testing"""


@work_in_tmp_dir()
@requires_working_xtb_install
def test_hessian_is_not_recalculated_if_present():
    mol = h2()
    xtb = XTB()

    optimiser = CartesianSDOptimiser(
        maxiter=1,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(1e-3),
    )
    optimiser.run(species=mol, method=xtb, n_cores=1)

    mol.calc_hessian(method=xtb)
    mol.hessian.__class__ = HessianInTesting

    # If the Hessian calculation is skipped then the class will be retained
    optimiser._update_hessian_gradient_and_energy()
    assert mol.hessian.__class__ == HessianInTesting


@work_in_tmp_dir()
@requires_working_xtb_install
def test_multiple_optimiser_saves_overrides_not_append():
    optimiser = CartesianSDOptimiser(
        maxiter=2,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(1e-3),
    )
    optimiser.run(method=XTB(), species=h2(), name="tmp.zip")

    assert os.path.isfile("tmp.zip")
    with zipfile.ZipFile("tmp.zip") as file:
        names = file.namelist()

    old_n_coords = sum([1 for name in names if name.startswith("coords_")])

    optimiser = CartesianSDOptimiser(
        maxiter=2,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(1e-3),
    )
    optimiser.run(method=XTB(), species=h2(), name="tmp.zip")
    # the file "tmp.zip" should be overwritten by new optimiser
    with zipfile.ZipFile("tmp.zip") as file:
        names = file.namelist()

    n_coords = sum([1 for name in names if name.startswith("coords_")])
    assert old_n_coords == n_coords


@work_in_tmp_dir()
def test_optimiser_print_geometries(caplog):
    mol = Molecule(smiles="C=C", name="mymolecule")
    coords1 = CartesianCoordinates(mol.coordinates)
    opt = CartesianSDOptimiser(maxiter=20, gtol=1e-3, etol=1e-4)
    opt._coords = coords1
    # cannot print geom without species
    with pytest.raises(AssertionError):
        opt.print_geometries()

    opt._species = mol
    assert opt.iteration == 0
    with caplog.at_level("WARNING"):
        opt.print_geometries()
    assert "Optimiser did no steps, not saving .xyz" in caplog.text
    assert not os.path.isfile("mymolecule_opt.trj.xyz")
    opt._coords = coords1.copy()
    opt.print_geometries()
    assert os.path.isfile("mymolecule_opt.trj.xyz")
    old_size = os.path.getsize("mymolecule_opt.trj.xyz")
    # running should overwrite the geometries
    opt.print_geometries()
    new_size = os.path.getsize("mymolecule_opt.trj.xyz")
    assert old_size == new_size


def _get_4_random_coordinates():
    coords_list = []
    for _ in range(4):
        coords_list.append(CartesianCoordinates(np.random.rand(6)))
    return coords_list


@work_in_tmp_dir()
def test_optimiser_history_storage():
    coords1, coords2, coords3, coords4 = _get_4_random_coordinates()

    hist = OptimiserHistory(maxlen=3)
    # cannot close without opening a file
    with pytest.raises(RuntimeError):
        hist.close()
    hist.open("test.zip")
    assert os.path.isfile("test.zip")
    # cannot reinitialise
    with pytest.raises(RuntimeError, match="cannot initialise again"):
        hist.open("test.zip")
    hist.add(coords1)
    hist.add(coords2)
    hist.add(coords3)
    # nothing should be on disk yet
    assert len(hist) == 3 and hist._n_stored == 0
    # now last coord is put on disk
    hist.add(coords4)
    assert len(hist) == 4 and hist._n_stored == 1
    assert len(hist._memory) == 3
    hist.close()
    # now should be 3 more stored on disk
    assert len(hist) == 4 and hist._n_stored == 4
    # adding new coords is forbidden
    with pytest.raises(RuntimeError):
        hist.add(coords1)
    # iterate through the history in reverse
    iterator = reversed(hist)
    last = next(iterator)
    before_last = next(iterator)
    assert np.allclose(last, coords4) and np.allclose(before_last, coords3)
    # clean up
    hist.clean_up()
    assert not os.path.isfile("test.zip")


@work_in_tmp_dir()
def test_optimiser_history_getitem():
    coords0, coords1, coords2, coords3 = _get_4_random_coordinates()
    hist = OptimiserHistory(maxlen=2)
    hist.open("test.zip")
    hist.add(coords0)
    hist.add(coords1)
    hist.add(coords2)
    hist.add(coords3)
    assert np.allclose(hist[0], coords0)  # from disk
    hist[0].e = PotentialEnergy(0.001, "Ha")
    assert hist[0].e is None  # cannot modify disk
    assert np.allclose(hist[2], coords2)  # from memory
    assert hist[2].e is None
    hist[2].e = PotentialEnergy(0.01, "Ha")
    assert np.isclose(hist[2].e, 0.01)
    # slicing does not work
    with pytest.raises(NotImplementedError):
        _ = hist[0:1]
    # can only have integer indices
    with pytest.raises(ValueError):
        _ = hist["x"]
    with pytest.raises(IndexError):
        _ = hist[4]
    with pytest.raises(IndexError):
        _ = hist[-5]
    # if no disk backend, then old coordinates are lost
    hist_nodisk = OptimiserHistory(maxlen=2)
    hist_nodisk.add(coords0)
    hist_nodisk.add(coords1)
    hist_nodisk.add(coords2)
    assert hist_nodisk[0] is None
    assert hist_nodisk._n_stored == 0


@work_in_tmp_dir()
def test_optimiser_history_reload():
    coords0, coords1, coords2, coords3 = _get_4_random_coordinates()
    hist = OptimiserHistory(maxlen=2)
    hist.open("savefile")
    assert os.path.isfile("savefile.zip")  # extension added
    hist.add(coords0)
    hist.add(coords1)
    hist.add(coords2)
    hist.add(coords3)
    hist.close()
    hist = None
    with pytest.raises(FileNotFoundError, match="test.zip does not exist"):
        _ = OptimiserHistory.load("test")
    with open("test", "w") as fh:
        fh.write("abcd")
    # error if file is not zip
    with pytest.raises(ValueError, match="not a valid trajectory"):
        _ = OptimiserHistory.load("test")
    # error if file does not have the autodE opt header
    with zipfile.ZipFile("new.zip", "w") as file:
        file.write("test")
    with pytest.raises(ValueError, match="not an autodE trajectory"):
        _ = OptimiserHistory.load("new.zip")
    hist = OptimiserHistory.load("savefile")
    assert np.allclose(hist[-1], coords3)
    assert np.allclose(hist[-2], coords2)
    assert np.allclose(hist[-3], coords1)


@work_in_tmp_dir()
def test_optimiser_history_reload_works_with_one():
    coords0 = CartesianCoordinates(np.random.rand(6))
    hist = OptimiserHistory(maxlen=2)
    hist.open("savefile")
    # adding None will not do anything
    hist.add(None)
    assert len(hist) == 0
    # just add one more coordinate
    hist.add(coords0)
    hist.close()
    assert os.path.isfile("savefile.zip")
    hist = OptimiserHistory.load("savefile")
    assert len(hist) == 1
    assert np.allclose(hist[0], coords0)
    assert hist[0] is hist[-1]


def test_mocked_method():
    method = Method()
    assert method.implements(CalculationType.energy)
    assert repr(method) is not None  # just needs to be implemented


def test_null_optimiser_methods():
    optimiser = NullOptimiser()
    optimiser.run()
    # run does nothing

    with pytest.raises(RuntimeError):
        _ = optimiser.final_coordinates
