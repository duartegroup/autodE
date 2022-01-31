import pytest
import numpy as np
from autode import Molecule, Atom
from autode.methods import XTB
from autode.values import GradientNorm, PotentialEnergy
from autode.utils import work_in_tmp_dir
from ..testutils import requires_with_working_xtb_install
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.steepest_decent import (CartesianSDOptimiser,
                                                   DIC_SD_Optimiser)


def methane_mol():
    return Molecule(atoms=[Atom('C',  0.11105, -0.21307,  0.00000),
                           Atom('H',  1.18105, -0.21307,  0.00000),
                           Atom('H', -0.24562, -0.89375,  0.74456),
                           Atom('H', -0.24562, -0.51754, -0.96176),
                           Atom('H', -0.24562,  0.77207,  0.21720)])


def h2():
    return Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.5)])


def sample_cartesian_optimiser():
    return CartesianSDOptimiser(maxiter=1,
                                gtol=GradientNorm(0.1),
                                etol=PotentialEnergy(0.1))


def test_optimiser_construct():

    # Optimiser needs a Species
    with pytest.raises(ValueError):
        sample_cartesian_optimiser().run(species=None, method=XTB())

    # also a method
    with pytest.raises(ValueError):
        sample_cartesian_optimiser().run(species=methane_mol(), method=None)

    # Optimiser needs valid arguments
    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(maxiter=0,
                                 gtol=GradientNorm(0.1),
                                 etol=PotentialEnergy(0.1))

    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(maxiter=1,
                                 gtol=GradientNorm(-0.1),
                                 etol=PotentialEnergy(0.1))

    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(maxiter=1,
                                 gtol=GradientNorm(-0.1),
                                 etol=PotentialEnergy(0.1))

    with pytest.raises(ValueError):
        _ = CartesianSDOptimiser(maxiter=1,
                                 gtol=GradientNorm(0.1),
                                 etol=PotentialEnergy(-0.1))


def test_initialise_species_and_method():

    optimiser = sample_cartesian_optimiser()

    # Species and method need to be valid
    with pytest.raises(ValueError):
        optimiser._initialise_species_and_method(species=None,
                                                 method=None)

    with pytest.raises(ValueError):
        optimiser._initialise_species_and_method(species='a',
                                                 method=None)


def test_coords_set():

    optimiser = sample_cartesian_optimiser()

    # Internal set of coordinates must be an instance of OptCoordinate
    with pytest.raises(ValueError):
        optimiser._coords = 'a'


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

    assert np.isclose(diff_e, 0.1, atol=1E-6)


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
    assert not optimiser._history.contains_well


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

    optimiser = CartesianSDOptimiser(maxiter=2,
                                     gtol=GradientNorm(0.01),
                                     etol=PotentialEnergy(1E-3),
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
    optimiser = DIC_SD_Optimiser(step_size=2.5,
                                 maxiter=10,
                                 gtol=GradientNorm(0.01),
                                 etol=PotentialEnergy(0.0001))

    mol = h2()
    # Should optimise fast, in only a few steps
    optimiser.run(species=mol, method=XTB())

    assert optimiser.converged
    assert optimiser.iteration < 10
    assert np.isclose(mol.distance(0, 1), 0.77, atol=0.1)