import pytest
import numpy as np

from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.values import Distance
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.utils import work_in_tmp_dir
from ..testutils import requires_with_working_xtb_install


def crfo_coords(molecule):
    optimiser = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5)
    optimiser._species = molecule
    optimiser._build_internal_coordinates()

    return optimiser._coords


def _tmp():
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

        print(coords.g)

        # dL/dλ = C(x) = (r_OH^-1 - r_ideal^-1)
        assert np.isclose(coords.g[3], coords.primitives[0].delta(coords._x))


    def test_simple_hessian_update():

        coords = crfo_water_coords()

        cartesian_h = 9.9 * np.eye(9)
        coords.update_h_from_cart_h(cartesian_h)

        assert coords.h.shape == (4, 4)

        assert not np.isclose(coords.h[0, 0], 0.)  # d^2L/ds_0ds_0
        assert not np.isclose(coords.h[0, 1], 0.)  # d^2L/ds_0ds_1

        assert np.isclose(coords.h[3, 3], 0.)  # d^2L/dλ^2
        assert np.isclose(coords.h[2, 3], 1.)  # d^2L/dλ^2

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

        optimiser = CRFOptimiser(etol=1, gtol=1, maxiter=1)
        empty_step = np.array([])
        assert len(optimiser._sanitised_step(empty_step)) == 0


    @requires_with_working_xtb_install
    @work_in_tmp_dir()
    def test_xtb_opt_with_distance_constraint():

        water = Molecule(name='water', charge=0, mult=1,
                         atoms=[Atom('O', -0.00110, 0.36310,  0.00000),
                                Atom('H', -0.82500, -0.18190, 0.00000),
                                Atom('H',  0.82610, -0.18120, 0.00000)])

        water.constraints.distance = {(0, 1): Distance(1.1, units='Å')}

        assert np.isclose(water.distance(0, 1),
                          0.99,
                          atol=0.01)

        CRFOptimiser.optimise(species=water, method=XTB())

        # Optimisation should generate an O-H distance *very* close to 1.1 Å
        assert np.isclose(water.distance(0, 1).to('Å'),
                          1.1,
                          atol=1E-4)


def test_step_c4h6():

    m = Molecule(atoms=[
        Atom('C', -1.74017,  0.51876, -0.4458),
        Atom('C', -0.46175,  0.54316, -0.0858),
        Atom('C',  0.46171, -0.54342, -0.3310),
        Atom('C',  1.74009, -0.51866,  0.0290),
        Atom('H', -2.16714, -0.33204, -0.9517),
        Atom('H', -2.40540,  1.34232, -0.2511),
        Atom('H', -0.05408,  1.40859,  0.4209),
        Atom('H',  0.05430, -1.40899, -0.8378),
        Atom('H',  2.16678,  0.33235,  0.5350),
        Atom('H',  2.40567, -1.34207, -0.1653),
    ])
    m.constraints.distance = {(0, 1): m.distance(0, 1)+0.1}

    coords = crfo_coords(m)
    s = np.zeros_like(coords)
    for i in range(24):
        s[i] = 0.01
        _ = coords.copy() + s
        s[i] = 0.

    # coords += np.random.uniform(-0.01, 0.01, size=coords.shape)
    print(np.round(coords.U, 3))
    print(coords.U.shape)

    print(np.linalg.matrix_rank(coords.U))



