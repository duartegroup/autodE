import os
import pytest
import numpy as np

from autode.methods import XTB
from autode.values import GradientRMS, PotentialEnergy
from autode.hessians import Hessian
from autode.utils import work_in_tmp_dir
from ..testutils import requires_with_working_xtb_install
from .molecules import h2, methane_mol, h_atom
from .setup import Method
from autode.utils import NumericStringDict
from autode.opt.coordinates import CartesianCoordinates
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
    optimiser._history.append(CartesianCoordinates([0.0, 1.0]))
    optimiser._history.append(CartesianCoordinates([0.0, 1.1]))

    # 2nd iteration for a history of two, indexed from 0
    assert optimiser.iteration == 1

    # without defined energies |E_0 - E_1| cannot be calculated
    with pytest.raises(RuntimeError):
        _ = optimiser._abs_delta_e

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

    optimiser._history.append(c1)

    c2 = CartesianCoordinates([1.1, 0.0, 0.0])
    c2.h = np.eye(3)

    optimiser._history.append(c2)

    # and try and update the (inverse) hessian, which is impossible without
    # an updater
    with pytest.raises(RuntimeError):
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

    # or minimum in energy
    with pytest.raises(IndexError):
        _ = optimiser._history.minimum

    # and cannot contain a well in the energy
    assert not optimiser._history.contains_energy_rise


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_xtb_h2_cart_opt():

    mol = h2()
    CartesianSDOptimiser.optimise(mol, method=XTB(), maxiter=50)

    # Optimised H-H distance is ~0.7 Å
    assert np.isclose(mol.distance(0, 1), 0.777, atol=0.1)


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_xtb_h2_cart_opt():

    optimiser = CartesianSDOptimiser(
        maxiter=2,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(1e-3),
    )
    assert not optimiser.converged
    optimiser._species = h2()

    assert not optimiser.converged

    # Should not converge in only two steps
    optimiser.run(method=XTB(), species=h2())
    assert not optimiser.converged


@work_in_tmp_dir()
@requires_with_working_xtb_install
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
@requires_with_working_xtb_install
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
@requires_with_working_xtb_install
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
@requires_with_working_xtb_install
def test_multiple_optimiser_saves_overrides_not_append():

    optimiser = CartesianSDOptimiser(
        maxiter=2,
        gtol=GradientRMS(0.01),
        etol=PotentialEnergy(1e-3),
    )
    optimiser.run(method=XTB(), species=h2())
    optimiser.save("tmp.traj")

    def n_lines_in_traj_file():
        return len(open("tmp.traj", "r").readlines())

    n_init_lines = n_lines_in_traj_file()
    optimiser.save("tmp.traj")

    assert n_lines_in_traj_file() == n_init_lines
