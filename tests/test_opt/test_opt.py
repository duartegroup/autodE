import pytest
import shutil
import numpy as np
from autode import Molecule, Atom
from autode.methods import XTB
from autode.values import GradientNorm, PotentialEnergy
from autode.utils import work_in_tmp_dir
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


@work_in_tmp_dir()
def test_xtb_h2_cart_opt():

    # Don't run the calculation without a working XTB install
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    mol = h2()
    CartesianSDOptimiser.optimise(mol, method=XTB(), maxiter=50)

    # Optimised H-H distance is ~0.7 Å
    assert np.isclose(mol.distance(0, 1), 0.777, atol=0.1)


@work_in_tmp_dir()
def test_xtb_h2_cart_opt():

    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    optimiser = CartesianSDOptimiser(maxiter=2,
                                     gtol=GradientNorm(0.1),
                                     etol=PotentialEnergy(0.1),
                                     )
    assert not optimiser.converged
    optimiser._species = h2()

    assert not optimiser.converged

    # Should not converge in only two steps
    optimiser.run(method=XTB(), species=h2())
    assert not optimiser.converged


@work_in_tmp_dir()
def test_xtb_h2_dic_opt():

    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

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
