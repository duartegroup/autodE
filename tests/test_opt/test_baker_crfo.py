import pytest
import numpy as np

from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.values import Distance
from autode.opt.optimisers.baker_crfo import CRFOptimiser


def water_molecule(oh_distance=1):
    """Water molecule with a constraint"""

    m = Molecule(name='water', charge=0, mult=1,
                 atoms=[Atom('O', -0.00110, 0.36310, 0.00000),
                        Atom('H', -0.82500, -0.18190, 0.00000),
                        Atom('H', 0.82610, -0.18120, 0.00000)])
    m.constraints.distance = {(0, 1): Distance(oh_distance, "Å")}

    return m


def test_coordinate_setup():

    mol = water_molecule()
    dist_consts = mol.constraints.distance
    assert (0, 1) in dist_consts and (1, 0) in dist_consts

    opt = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5)

    with pytest.raises(RuntimeError):
        # Cannot set coordinates without a species
        opt._set_initial_coordinates()

    opt._species = mol
    opt._set_initial_coordinates()
    assert opt._coords.n_constraints == 1

    # Ensure that the final DIC comprises a single primitive, which is the
    # first (inverse) distance populated in the coordinates
    assert np.allclose(opt._coords.U[:, 2],
                       np.array([1., 0., 0.]))

    # Initial lagrangian multiplier is close to zero, which is the last
    # component in the optimisation space
    opt._coords.zero_lagrangian_multipliers()
    assert np.isclose(opt._coords[2], 0.)


def crfo_water_coords():

    optimiser = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5)
    optimiser._species = water_molecule()
    optimiser._set_initial_coordinates()

    return optimiser._coords


def test_gradient_update():

    coords = crfo_water_coords()

    cartesian_g = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    coords.update_g_from_cart_g(cartesian_g)

    # dL/dλ = C(x) = (r_OH^-1 - r_ideal^-1)
    assert np.isclose(coords.g[2], coords.primitives[0].delta(coords._x))
