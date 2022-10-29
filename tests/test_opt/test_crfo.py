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
from autode.opt.coordinates.primitives import DihedralAngle
from autode.utils import work_in_tmp_dir
from .molecules import h2o2_mol
from ..testutils import requires_with_working_xtb_install


def crfo_coords(molecule):
    optimiser = CRFOptimiser(maxiter=1, gtol=1e-5, etol=1e-5)
    optimiser._species = molecule
    optimiser._build_internal_coordinates()

    return optimiser._coords


def water_molecule(oh_distance=1):
    """Water molecule with a constraint"""

    m = Molecule(
        name="water",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", -0.00110, 0.36310, 0.00000),
            Atom("H", -0.82500, -0.18190, 0.00000),
            Atom("H", 0.82610, -0.18120, 0.00000),
        ],
    )
    m.constraints.distance = {(0, 1): val.Distance(oh_distance, "Å")}

    return m


def test_coordinate_setup():

    mol = water_molecule()
    dist_consts = mol.constraints.distance
    assert (0, 1) in dist_consts and (1, 0) in dist_consts

    opt = CRFOptimiser(maxiter=1, gtol=1e-5, etol=1e-5)

    with pytest.raises(RuntimeError):
        # Cannot set coordinates without a species
        opt._build_internal_coordinates()

    opt._species = mol
    opt._build_internal_coordinates()
    assert opt._coords.n_constraints == 1

    # Ensure that the final DIC comprises a single primitive, which is the
    # first (inverse) distance populated in the coordinates
    assert np.allclose(opt._coords.U[:, 2], np.array([1.0, 0.0, 0.0]))

    # Initial lagrangian multiplier is close to zero, which is the last
    # component in the optimisation space
    opt._coords.zero_lagrangian_multipliers()
    assert np.isclose(opt._coords._lambda[0], 0.0)


def crfo_water_coords():
    return crfo_coords(molecule=water_molecule())


def test_setting_invalid_lagrange_multipliers():

    s = crfo_water_coords()
    # the vector of lagrange multipliers must be of length one to be
    # appropriate for a water molecule with a single constrained distance
    invalid_multipliers = np.ones(shape=(4,))

    with pytest.raises(ValueError):
        s.update_lagrange_multipliers(invalid_multipliers)


def test_simple_gradient_update():

    coords = crfo_water_coords()

    cartesian_g = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    coords.update_g_from_cart_g(cartesian_g)

    # dL/dλ = -C(x) = -(r_OH^-1 - r_ideal^-1)
    assert np.isclose(coords.g[3], -coords.primitives[0].delta(coords._x))


def test_simple_hessian_update():

    coords = crfo_water_coords()

    cartesian_h = 9.9 * np.eye(9)
    coords.update_h_from_cart_h(cartesian_h)

    assert coords.h.shape == (4, 4)

    assert not np.isclose(coords.h[0, 0], 0.0)  # d^2L/ds_0ds_0
    assert not np.isclose(coords.h[0, 1], 0.0)  # d^2L/ds_0ds_1

    assert np.isclose(coords.h[3, 3], 0.0)  # d^2L/dλ^2
    assert np.isclose(coords.h[2, 3], -1.0)  # d^2L/dλ^2

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
    s += np.array([0.03, -0.07, 0.0])

    def r(_x):
        return np.linalg.norm(_x[:3] - _x[3:6])

    # Should not change value of the 'removed' coordinate
    assert np.isclose(r(s.to("cartesian")), r_initial, atol=1e-10)


def test_init_g_norm_is_none():

    optimiser = CRFOptimiser(etol=1, gtol=1, maxiter=1)
    assert optimiser._g_norm > 0


def test_sanitised_zero_length_step():
    """Should be able to update with a null step"""

    optimiser = CRFOptimiser(etol=1, gtol=1, maxiter=1)
    optimiser._take_step_within_trust_radius(np.array([]))


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_opt_with_distance_constraint():

    water = Molecule(
        name="water",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", -0.00110, 0.36310, 0.00000),
            Atom("H", -0.82500, -0.18190, 0.00000),
            Atom("H", 0.82610, -0.18120, 0.00000),
        ],
    )

    water.constraints.distance = {(0, 1): val.Distance(1.1, units="Å")}

    assert np.isclose(water.distance(0, 1), 0.99, atol=0.01)

    CRFOptimiser.optimise(species=water, method=XTB())

    # Optimisation should generate an O-H distance *very* close to 1.1 Å
    assert np.isclose(water.distance(0, 1).to("Å"), 1.1, atol=1e-4)


def test_step_c2h3():

    m = Molecule(
        atoms=[
            Atom("C", 0.63365, 0.11934, -0.13163),
            Atom("C", -0.63367, -0.11938, 0.13153),
            Atom("H", 1.28230, -0.63391, -0.54779),
            Atom("H", 1.08517, 1.07993, 0.05600),
            Atom("H", -1.08517, -1.07984, -0.05599),
        ]
    )
    m.constraints.distance = {(0, 1): m.distance(0, 1)}

    coords = crfo_coords(m)

    # Should be able to add an arbitrary vector to the coordinates
    coords += np.random.uniform(-0.1, 0.1, size=coords.shape)


def test_baker1997_example():

    c2h3f = Molecule(
        atoms=[
            Atom("C", 0.061684, 0.673790, 0.0),
            Atom("C", 0.061684, 0.726210, 0.0),
            Atom("F", 1.174443, 1.331050, 0.0),
            Atom("H", 0.927709, 1.173790, 0.0),
            Atom("H", 0.927709, 1.226210, 0.0),
            Atom("H", 0.804342, 1.226210, 0.0),
        ]
    )

    r1 = prim.ConstrainedDistance(0, 1, value=1.5)
    r2 = prim.ConstrainedDistance(3, 4, value=2.5)
    theta = prim.ConstrainedBondAngle(
        1, 0, 5, value=val.Angle(123.0, "º").to("rad")
    )

    pic = PIC(r1, r2, theta)
    for pair in ((2, 0), (3, 0), (4, 1), (5, 1)):
        pic.append(prim.Distance(*pair))

    for triple in (
        (1, 0, 3),
        (1, 0, 3),
        (2, 0, 3),
        (4, 1, 0),
        (5, 1, 0),
        (4, 1, 5),
    ):
        pic.append(prim.BondAngle(*triple))

    for quadruple in ((4, 1, 0, 2), (4, 1, 0, 3), (5, 1, 0, 2), (5, 1, 0, 3)):
        pic.append(prim.DihedralAngle(*quadruple))

    dic = DICWithConstraints.from_cartesian(
        x=CartesianCoordinates(c2h3f.coordinates), primitives=pic
    )
    assert len(dic) == 10


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_crfo_with_dihedral():

    mol = h2o2_mol()
    constrained_distance = mol.distance(0, 1) + 0.1
    mol.constraints.distance = {(0, 1): constrained_distance}

    CRFOptimiser.optimise(species=mol, method=XTB(), maxiter=10)

    assert np.isclose(mol.distance(0, 1), constrained_distance, atol=1e-4)


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_opt_with_two_distance_constraint():

    water = Molecule(
        name="water",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", -0.00110, 0.36310, 0.00000),
            Atom("H", -0.82500, -0.18190, 0.00000),
            Atom("H", 0.82610, -0.18120, 0.00000),
        ],
    )

    water.constraints.distance = {
        (0, 1): val.Distance(1.0, units="Å"),
        (0, 2): val.Distance(1.0, units="Å"),
    }

    opt = CRFOptimiser(maxiter=1, gtol=1e-5, etol=1e-5, init_alpha=0.1)
    opt._species = water
    opt._method = XTB()
    opt._initialise_run()

    # Moving the angle should preserve the distances
    s = opt._coords + np.array([0.1, 0.0, 0.0])
    x = s.to("cart").reshape((3, 3))
    for (i, j) in ((0, 1), (0, 2)):
        assert np.isclose(
            np.linalg.norm(x[i, :] - x[j, :]), water.distance(i, j)
        )

    opt._step()
    x = opt._coords.to("cart").reshape((3, 3))
    for (i, j) in ((0, 1), (0, 2)):
        assert np.abs(np.linalg.norm(x[i, :] - x[j, :]) - 1.0) < np.abs(
            water.distance(i, j) - 1.0
        )

    opt = CRFOptimiser(maxiter=10, gtol=1e-5, etol=1e-5, init_alpha=0.1)
    opt.run(species=water, method=XTB())
    assert opt.converged

    for pair in ((0, 1), (0, 2)):
        assert np.isclose(water.distance(*pair).to("Å"), 1.0, atol=1e-2)


class DICEnsureBackTransform(DICWithConstraints):
    @property
    def _allow_unconverged_back_transform(self) -> bool:
        raise RuntimeError("Must converge back transformation")


def test_step_with_180degree_dihedrals():

    ethane = Molecule(
        atoms=[
            Atom("C", -6.05284, 0.86485, 0.00000),
            Atom("C", -4.56258, 1.15982, 0.00000),
            Atom("H", -6.48013, 1.08394, 1.00120),
            Atom("H", -6.22501, -0.20494, -0.24263),
            Atom("H", -6.56172, 1.49618, -0.75857),
            Atom("H", -4.05370, 0.52850, 0.75857),
            Atom("H", -4.39041, 2.22962, 0.24263),
            Atom("H", -4.13530, 0.94074, -1.00120),
        ]
    )

    ds = [
        -1.900e-04,
        1.00e-05,
        -1.90e-04,
        2.00e-05,
        2.000e-04,
        3.749e-02,
        -1.800e-04,
        -8.00e-05,
        1.80e-04,
        1.80e-04,
        1.340e-03,
        8.700e-04,
        -2.880e-03,
        -1.20e-04,
        -6.40e-04,
        -1.80e-04,
        3.300e-04,
        9.863e-02,
    ]

    # Should be able to take a step without any warnings
    dic = crfo_coords(ethane)
    dic.__class__ = DICEnsureBackTransform  # oh so hacky
    dic += ds


def test_linear_dihedrals_are_removed():

    allene = Molecule(
        atoms=[
            Atom("C", 0.35540, -0.20370, -0.44810),
            Atom("C", -0.37180, 0.21470, 0.40200),
            Atom("H", 1.01560, -0.60550, -1.23530),
            Atom("H", -0.99920, 0.59450, 1.15720),
        ]
    )

    dic = crfo_coords(allene)
    assert not any(isinstance(q, DihedralAngle) for q in dic.primitives)
