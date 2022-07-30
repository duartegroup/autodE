import pytest
import numpy as np

import autode.values as val
import autode.opt.coordinates.primitives as prim
from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.opt.coordinates.internals import PIC
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.utils import work_in_tmp_dir
from .molecules import h2o2_mol
from ..testutils import requires_with_working_xtb_install


def crfo_coords(molecule):
    optimiser = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5)
    optimiser._species = molecule
    optimiser._build_internal_coordinates()

    return optimiser._coords


def water_molecule(oh_distance=1):
    """Water molecule with a constraint"""

    m = Molecule(name='water', charge=0, mult=1,
                 atoms=[Atom('O', -0.00110, 0.36310, 0.00000),
                        Atom('H', -0.82500, -0.18190, 0.00000),
                        Atom('H', 0.82610, -0.18120, 0.00000)])
    m.constraints.distance = {(0, 1): val.Distance(oh_distance, "Å")}

    return m


def test_coordinate_setup():

    mol = water_molecule()
    dist_consts = mol.constraints.distance
    assert (0, 1) in dist_consts and (1, 0) in dist_consts

    opt = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5)

    with pytest.raises(RuntimeError):
        # Cannot set coordinates without a species
        opt._build_internal_coordinates()

    opt._species = mol
    opt._build_internal_coordinates()
    assert opt._coords.n_constraints == 1

    # Ensure that the final DIC comprises a single primitive, which is the
    # first (inverse) distance populated in the coordinates
    assert np.allclose(opt._coords.U[:, 2],
                       np.array([1., 0., 0.]))

    # Initial lagrangian multiplier is close to zero, which is the last
    # component in the optimisation space
    opt._coords.zero_lagrangian_multipliers()
    assert np.isclose(opt._coords._lambda[0], 0.)


def crfo_water_coords():
    return crfo_coords(molecule=water_molecule())


def test_setting_invalid_lagrange_multipliers():

    s = crfo_water_coords()
    invalid_multipliers = np.ones(shape=(4,))

    with pytest.raises(ValueError):
        s.set_lagrange_multipliers(invalid_multipliers)


def test_simple_gradient_update():

    coords = crfo_water_coords()

    cartesian_g = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    coords.update_g_from_cart_g(cartesian_g)

    # dL/dλ = -C(x) = -(r_OH^-1 - r_ideal^-1)
    assert np.isclose(coords.g[3], -coords.primitives[0].delta(coords._x))


def test_simple_hessian_update():

    coords = crfo_water_coords()

    cartesian_h = 9.9 * np.eye(9)
    coords.update_h_from_cart_h(cartesian_h)

    assert coords.h.shape == (4, 4)

    assert not np.isclose(coords.h[0, 0], 0.)  # d^2L/ds_0ds_0
    assert not np.isclose(coords.h[0, 1], 0.)  # d^2L/ds_0ds_1

    assert np.isclose(coords.h[3, 3], 0.)  # d^2L/dλ^2
    assert np.isclose(coords.h[2, 3], -1.)  # d^2L/dλ^2

    # Hessian needs to be symmetric
    assert np.allclose(coords.h.T, coords.h)


def test_primitive_projection_discard():

    optimiser = CRFOptimiser(etol=1, gtol=1, maxiter=1)
    optimiser._species = water_molecule()

    # Current distance that will be constrained
    r_initial = optimiser._species.distance(0, 1)

    x = CartesianCoordinates(optimiser._species.coordinates)
    s = DICWithConstraints.from_cartesian(x, optimiser._primitives)
    assert len(s) == 3

    # Shift on the first couple of DIC but nothing on the final one
    s += np.array([0.03, -0.07, 0.])

    def r(_x):
        return np.linalg.norm(_x[:3] - _x[3:6])

    # Should not change value of the 'removed' coordinate
    assert np.isclose(r(s.to("cartesian")), r_initial, atol=1E-10)


def test_init_g_norm_is_none():

    optimiser = CRFOptimiser(etol=1, gtol=1, maxiter=1)
    assert optimiser._g_norm > 0


def test_sanitised_zero_length_step():
    """Should be able to update with a null step"""

    optimiser = CRFOptimiser(etol=1, gtol=1, maxiter=1)
    optimiser._take_sanitised_step(np.array([]))


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_opt_with_distance_constraint():

    water = Molecule(name='water', charge=0, mult=1,
                     atoms=[Atom('O', -0.00110, 0.36310,  0.00000),
                            Atom('H', -0.82500, -0.18190, 0.00000),
                            Atom('H',  0.82610, -0.18120, 0.00000)])

    water.constraints.distance = {(0, 1): val.Distance(1.1, units='Å')}

    assert np.isclose(water.distance(0, 1),
                      0.99,
                      atol=0.01)

    CRFOptimiser.optimise(species=water, method=XTB())

    # Optimisation should generate an O-H distance *very* close to 1.1 Å
    assert np.isclose(water.distance(0, 1).to('Å'),
                      1.1,
                      atol=1E-4)


def test_step_c2h3():

    m = Molecule(atoms=[
        Atom('C',  0.63365,  0.11934, -0.13163),
        Atom('C', -0.63367, -0.11938,  0.13153),
        Atom('H',  1.28230, -0.63391, -0.54779),
        Atom('H',  1.08517,  1.07993,  0.05600),
        Atom('H', -1.08517, -1.07984, -0.05599),
    ])
    m.constraints.distance = {(0, 1): m.distance(0, 1)}

    coords = crfo_coords(m)

    # Should be able to add an arbitrary vector to the coordinates
    coords += np.random.uniform(-0.1, 0.1, size=coords.shape)


def test_baker1997_example():

    c2h3f = Molecule(atoms=[
       Atom('C',  .061684,  .673790, .0),
       Atom('C',  .061684,  .726210, .0),
       Atom('F', 1.174443, 1.331050, .0),
       Atom('H',  .927709, 1.173790, .0),
       Atom('H',  .927709, 1.226210, .0),
       Atom('H',  .804342, 1.226210, .0),
    ])

    r1 = prim.ConstrainedDistance(0, 1, value=1.5)
    r2 = prim.ConstrainedDistance(3, 4, value=2.5)
    theta = prim.ConstrainedBondAngle(1, 0, 5,
                                      value=val.Angle(123.0, 'º').to('rad'))

    pic = PIC(r1, r2, theta)
    for pair in ((2, 0), (3, 0), (4, 1), (5, 1)):
        pic.append(prim.Distance(*pair))

    for triple in ((1, 0, 3), (1, 0, 3), (2, 0, 3),
                   (4, 1, 0), (5, 1, 0), (4, 1, 5)):
        pic.append(prim.BondAngle(*triple))

    for quadruple in ((4, 1, 0, 2), (4, 1, 0, 3), (5, 1, 0, 2), (5, 1, 0, 3)):
        pic.append(prim.DihedralAngle(*quadruple))

    dic = DICWithConstraints.from_cartesian(
        x=CartesianCoordinates(c2h3f.coordinates),
        primitives=pic
    )
    assert len(dic) == 10


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_crfo_with_dihedral():

    mol = h2o2_mol()
    constrained_distance = mol.distance(0, 1) + 0.1
    mol.constraints.distance = {(0, 1): constrained_distance}

    CRFOptimiser.optimise(species=mol, method=XTB(), maxiter=10)

    assert np.isclose(mol.distance(0, 1),
                      constrained_distance,
                      atol=1E-4)


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_opt_with_two_distance_constraint():

    water = Molecule(name='water', charge=0, mult=1,
                     atoms=[Atom('O', -0.00110, 0.36310,  0.00000),
                            Atom('H', -0.82500, -0.18190, 0.00000),
                            Atom('H',  0.82610, -0.18120, 0.00000)])

    water.constraints.distance = {
            (0, 1): val.Distance(1.0, units='Å'),
            (0, 2): val.Distance(1.0, units='Å'),
            }

    opt = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5, init_alpha=0.1)
    opt._species = water
    opt._method = XTB()
    opt._initialise_run()

    # Moving the angle should preserve the distances
    s = opt._coords + np.array([0.1, 0.0, 0.0])
    x = s.to("cart").reshape((3, 3))
    for (i, j) in ((0, 1), (0, 2)):
        assert np.isclose(
                np.linalg.norm(x[i, :] - x[j, :]),
                water.distance(i, j)
                )

    opt._step()
    x = opt._coords.to("cart").reshape((3, 3))
    for (i, j) in ((0, 1), (0, 2)):
        assert (np.abs(np.linalg.norm(x[i, :] - x[j, :]) - 1.0)
                < np.abs(water.distance(i, j) - 1.0))

    opt = CRFOptimiser(maxiter=10, gtol=1E-5, etol=1E-5, init_alpha=0.1)
    opt.run(species=water, method=XTB())
    assert opt.converged

    for pair in ((0, 1), (0, 2)):
        assert np.isclose(water.distance(*pair).to('Å'), 1.0, atol=1E-2)
