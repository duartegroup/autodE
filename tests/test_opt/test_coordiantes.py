import itertools

import pytest
import numpy as np
from .molecules import (
    h2,
    methane_mol,
    water_mol,
    h2o2_mol,
    feco5_mol,
    cumulene_mol,
    acetylene_mol,
)
from autode.utils import work_in_tmp_dir
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.values import Angle
from autode.exceptions import CoordinateTransformFailed
from autode.opt.coordinates.internals import (
    PrimitiveInverseDistances,
    PIC,
    AnyPIC,
)
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.coordinates.dic import DIC, DICWithConstraints
from autode.opt.coordinates.primitives import (
    PrimitiveInverseDistance,
    PrimitiveDistance,
    ConstrainedPrimitiveDistance,
    PrimitiveBondAngle,
    ConstrainedPrimitiveBondAngle,
    PrimitiveDihedralAngle,
    PrimitiveImproperDihedral,
    PrimitiveLinearAngle,
    PrimitiveDummyLinearAngle,
    LinearBendType,
    CompositeBonds,
    ConstrainedCompositeBonds,
)


def test_inv_dist_primitives():
    arr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    inv_dist = PrimitiveInverseDistance(0, 1)
    assert np.isclose(inv_dist(x), 0.5)  # 1/2.0 = 0.5 Å-1

    # Check a couple of derivatives by hand
    derivs = inv_dist.derivative(x=x)
    assert np.isclose(derivs[3 * 0 + 0], 2 * inv_dist(x) ** 3)
    assert np.isclose(derivs[3 * 1 + 0], -2 * inv_dist(x) ** 3)

    # Derivatives with respect to zero components
    assert np.isclose(derivs[3 * 0 + 1], 0)


def test_dist_primitives():
    arr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    inv_dist = PrimitiveDistance(0, 1)
    assert np.isclose(inv_dist(x), 2.0)

    derivs = inv_dist.derivative(x)
    assert np.isclose(derivs[3 * 0 + 0], -2 / 2)
    assert np.isclose(derivs[3 * 1 + 0], +2 / 2)

    for k in (1, 2):
        assert np.isclose(derivs[3 * 1 + k], 0)


def test_primitive_equality():
    assert PrimitiveInverseDistance(0, 1) != "a"
    assert PrimitiveInverseDistance(0, 1) == PrimitiveInverseDistance(0, 1)
    assert PrimitiveInverseDistance(1, 0) == PrimitiveInverseDistance(0, 1)


def test_primitives_equality():
    x = CartesianCoordinates(h2().coordinates)
    primitives = PrimitiveInverseDistances.from_cartesian(x)

    assert primitives != "a"
    assert primitives == PrimitiveInverseDistances.from_cartesian(x)

    # Order does not matter for equality
    assert primitives == PrimitiveInverseDistances(
        PrimitiveInverseDistance(1, 0)
    )


def test_cartesian_coordinates():
    arr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)  # defaults to Å
    assert "cart" in repr(x).lower()
    assert x.ndim == 1

    # Test equality
    assert x != "some string"
    assert x != arr

    # Can convert to other distance units
    assert 0.19 < CartesianCoordinates(arr).to("nm")[3] < 0.21

    # but not an unsupported unit
    with pytest.raises(ValueError):
        _ = CartesianCoordinates(arr).to("X")


def test_cartesian_coordinates_hessian_update():
    # Simple coordinates with 2 atoms in 3 D
    coords = CartesianCoordinates(np.arange(0, 6).reshape((2, 3)))

    with pytest.raises(AssertionError):
        coords.update_h_from_cart_h(arr=np.array([]))

    with pytest.raises(AssertionError):
        coords.update_h_from_cart_h(arr=np.array([1.0]))

    # Hessian needs to be 6x6
    coords.update_h_from_cart_h(arr=np.eye(6))
    assert coords.h is not None

    # Can set back to None
    coords.update_h_from_cart_h(arr=None)
    assert coords.h is None


def test_cartesian_coordinate_shift_type():
    coords = CartesianCoordinates(np.array([0.0]))

    # Shifting coordinates should retain the type
    assert isinstance(coords - 0.1, CartesianCoordinates)
    assert isinstance(0.1 - coords, CartesianCoordinates)
    assert isinstance(coords + 0.1, CartesianCoordinates)
    assert isinstance(0.1 + coords, CartesianCoordinates)


def test_hessian_set():
    coords = CartesianCoordinates(np.array([1.0, 2.0]))

    # Hessian and inverse must be NxN matrix, i.e. 2x2 here
    for invalid_h in (
        3,
        np.array([1.0]),
        np.arange(3),
        np.arange(9).reshape(3, 3),
        np.arange(6).reshape(2, 3),
    ):
        with pytest.raises(Exception):
            coords.h = invalid_h

        with pytest.raises(Exception):
            coords.h_inv = invalid_h


def test_hessian_inv():
    coords = CartesianCoordinates(np.array([1.0, 2.0]))
    coords.h = 2.0 * np.eye(2)

    expected_h_inv = 0.5 * np.eye(2)

    # Without setting the inverse Hessian it should be
    # calculable
    assert np.allclose(coords.h_inv, expected_h_inv, atol=1e-10)


def test_h_update():
    coords1 = CartesianCoordinates(np.array([1.0, 2.0]))
    coords1.g = np.array([0.1, 0.2])
    coords1.h = 2.0 * np.eye(2)

    coords2 = CartesianCoordinates(np.array([1.1, 2.1]))
    coords2.g = np.array([0.01, 0.02])
    assert coords2.h is None

    # must define a valid update type
    with pytest.raises(RuntimeError):
        coords2.update_h_from_old_h(coords1, hessian_update_types=[])

    from autode.opt.optimisers.hessian_update import BFGSSR1Update

    coords2.update_h_from_old_h(coords1, hessian_update_types=[BFGSSR1Update])
    assert coords2.h is not None


def test_cartesian_update_clear():
    arr = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    # Gradients and Hessians are initialized to None
    x = CartesianCoordinates(arr)
    assert x.g is None and x.h is None

    x.g = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    x.h = np.eye(6)

    # Perturbing the coordinates should clear the gradient, as it's no longer
    # accurate at these new coordinates
    x += np.arange(0, 6, dtype=float)
    assert x.g is None and x.h is None

    # or if the values are reassigned individually
    x.h = np.eye(6)
    x[0] = 0.1
    assert x.h is None

    # or totally with index slicing
    x.h = np.eye(6)
    x[:] = 0.0
    assert x.h is None


def test_basic_dic_properties():
    x = DIC(np.array([1.0]))
    assert "dic" in repr(x).lower()

    with pytest.raises(Exception):
        _ = x.to("unknown coordinates")


def test_dic_constraints():
    mol = water_mol()
    mol.constraints.distance = {(0, 1): 1.5}

    pic = AnyPIC.from_species(mol)
    x = CartesianCoordinates(mol.coordinates)
    q = DICWithConstraints.from_cartesian(x, pic)
    assert q.n_constraints == 1
    assert q.g is None and q.h is None

    # with constraints grad or hessian cannot be set directly
    with pytest.raises(RuntimeError):
        q.g = np.arange(3)
    with pytest.raises(RuntimeError):
        q.h = np.arange(3)

    q.update_g_from_cart_g(np.random.rand(9))
    q.update_h_from_cart_h(np.random.rand(9, 9))
    # one extra dimension from Lagrange multiplier
    assert q.g.shape == (4,)
    assert q.h.shape == (4, 4)


def test_invalid_pic_construction():
    # Cannot construct some primitives e.g. PrimitiveInverseDistances from non Primitive
    # internal coordinates
    with pytest.raises(ValueError):
        _ = PrimitiveInverseDistances("a")


def test_cart_to_dic():
    arr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    # Should only have 1 internal coordinate
    pic = PrimitiveInverseDistances.from_cartesian(x)
    assert len(pic) == 1

    # and not have a B matrix
    with pytest.raises(AttributeError):
        _ = pic.B

    # Delocalised internals should preserve the single internal coordinate
    dics = x.to("dic")
    assert isinstance(dics, DIC)
    assert len(dics) == 1
    # and store the previous catesian coordinates
    assert hasattr(dics, "_x")
    # as a copy, so changing the initial x should not change prev_x
    assert id(dics._x) != id(x)

    assert dics.inactive_indexes == []

    x += 0.1
    assert np.allclose(x, np.array([0.1, 0.1, 0.1, 2.1, 0.1, 0.1]))
    assert not np.allclose(x, dics._x)
    x -= 0.1


def test_cartesian_all_indexes_active():
    arr = np.arange(6)
    x = CartesianCoordinates(arr)
    assert x.active_indexes == list(range(6))
    assert x.inactive_indexes == list()


def test_simple_dic_to_cart():
    arr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Should be able to transform back to Cartesians
    dic = CartesianCoordinates(arr).to("dic")

    x = dic.to("cartesian")
    assert np.allclose(CartesianCoordinates(arr), x)

    delta = np.array([0.1])
    assert np.isclose(0.6, (dic + delta)[0], atol=1e-6)
    # Updating the DICs should afford cartesian coordinates that are
    # ~1.7 Å apart (1/r = 0.6)
    dic.iadd(value=delta)
    assert dic.shape == (1,)
    assert np.isclose(dic[0], 0.6)

    arr_update = dic.to("cart").reshape((2, 3))

    assert np.isclose(
        np.linalg.norm(arr_update[0, :] - arr_update[1, :]), 1.66666, atol=1e-4
    )


def test_methane_cart_to_dic():
    x = CartesianCoordinates(methane_mol().coordinates)
    dic = x.to("dic")
    assert len(dic) == 9  # 3N-6 for N=5

    dic.iadd(value=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Cartesian coordinates should be close to the starting ones
    assert np.linalg.norm(x - dic.to("cart")) < 0.5


def test_co2_cart_to_dic():
    arr = np.array(
        [
            [-1.31254, 0.34625, -0.00000],
            [-0.11672, 0.30964, 0.00000],
            [1.07904, 0.27311, 0.00000],
        ]
    )

    x = CartesianCoordinates(arr)
    dic = x.to("dic")
    assert len(dic) == 3


def test_grad_transform_linear():
    k = 1.0
    r0 = 1.0

    def energy(_x):
        """Harmonic potential: E = k(r-r0)^2"""
        _x = _x.reshape((-1, 3))
        r = np.linalg.norm(_x[0] - _x[1])
        return 0.5 * k * (r - r0) ** 2

    def grad(_x):
        _x = _x.reshape((-1, 3))
        diff = _x[0, 0] - _x[1, 0]
        r = np.linalg.norm(_x[0] - _x[1])
        return np.array(
            [
                [k * (r - r0) * diff / r, 0.0, 0.0],
                [-k * (r - r0) * diff / r, 0.0, 0.0],
            ]
        )

    def num_grad(_x, h=1e-8):
        _g = []
        for i in range(len(_x.flatten())):
            x_ph = np.array(_x, copy=True)
            x_ph[i] += h

            g_i = (energy(x_ph) - energy(_x)) / h
            _g.append(g_i)

        return np.array(_g).reshape((2, 3))

    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(coords)

    assert np.allclose(num_grad(x), grad(x))
    x.g = grad(coords)

    dic = x.to("dic")
    assert dic.shape == (1,)  # Only a single distance
    assert np.isclose(dic[0], 0.5, atol=1e-6)  # 1/r_012 = 0.5 Å

    assert dic.g.shape == (1,)  # dE/ds_i has only a single component

    # Determined by hand
    assert np.isclose(dic.g[0], -1 / 0.5**2 * grad(coords).flatten()[3])


def test_hess_transform_linear():
    k = 1.0
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    x = CartesianCoordinates(coords)

    def energy(_x, r0=1):
        """Harmonic potential: E = k(r-r0)^2"""
        _x = _x.reshape((-1, 3))
        r = np.linalg.norm(_x[0] - _x[1])
        return 0.5 * k * (r - r0) ** 2

    def hessian(_x):
        _x = _x.reshape((-1, 3))
        delta_x = _x[0, 0] - _x[1, 0]

        r = np.linalg.norm(_x[0] - _x[1])
        h_oo = k * (1 - 1 / r + delta_x**2 / r**3)

        _h = np.zeros(shape=(6, 6))
        _h[0, 0] = _h[3, 3] = h_oo
        _h[0, 3] = _h[3, 0] = -h_oo

        return _h

    def g_i(_x, i, h=1e-8):
        """Numerical graident"""
        x_ph = np.array(_x, copy=True)
        x_ph[i] += h
        return (energy(x_ph) - energy(_x)) / h

    def h_ij(_x, i, j, h=1e-8):
        x_ph = np.array(_x, copy=True)
        x_ph[j] += h

        return (g_i(x_ph, i=i) - g_i(_x, i=i)) / h

    def num_hess(_x):
        _h = np.zeros(shape=(6, 6))

        for i in range(6):
            for j in range(6):
                _h[i, j] = h_ij(_x, i, j)

        return _h

    assert np.linalg.norm(hessian(x) - num_hess(x)) < 1e-7
    x.h = hessian(coords)

    dic = x.to("dic")
    assert dic.h.shape == (1, 1)  # 1x1 internal Hessian
    assert np.isclose(dic.h[0, 0], k)  # should be ~k


def test_hess_positive_definite_no_hessian():
    """Cannot make a None Hessian positive definite"""

    coords = CartesianCoordinates(np.array([1.0, 2.0]))
    assert coords.h is None

    with pytest.raises(RuntimeError):
        coords.make_hessian_positive_definite()


def test_hess_positive_definite_sample():
    coords = CartesianCoordinates(np.array([1.0, 2.0]))
    coords.h = np.array([[1.0, 0.1], [0.1, -1.0]])

    # Have at least one negative eigenvalue
    assert sum(lmd < 0 for lmd in np.linalg.eigvalsh(coords.h)) > 0

    # Should have no negative eigenvalues
    coords.make_hessian_positive_definite()
    assert sum(lmd < 0 for lmd in np.linalg.eigvalsh(coords.h)) == 0


def test_hess_positive_definite_h2o():
    # h2o = Molecule('water',
    #                atoms=[Atom('O'),
    #                       Atom('H', -0.8, 0.1),
    #                       Atom('H', 0.8, 0.1)])

    coords_arr = np.array(
        [
            [0.0, -0.01118983, 0.0],
            [-0.8, 0.08881017, 0.0],
            [0.8, 0.08881017, 0.0],
        ]
    ).flatten()

    coords = CartesianCoordinates(coords_arr)

    coords.g = np.array(
        [
            [7.55890751e-09, 1.78450460e-01, -5.66918063e-09],
            [4.77548276e-01, -8.92130384e-02, 0.00000000e00],
            [-4.77548270e-01, -8.92130384e-02, 0.00000000e00],
        ]
    ).flatten()

    coords.h = np.array(
        [
            [
                11.2889,
                -0.0,
                -0.0,
                -5.6443,
                0.8169,
                -0.0,
                -5.6443,
                -0.8169,
                0.0,
            ],
            [-0.0, -1.528, -0.0, 0.7692, 0.764, 0.0, -0.7692, 0.764, 0.0],
            [-0.0, -0.0, -1.783, -0.0, 0.0, 0.8915, 0.0, 0.0, 0.8915],
            [
                -5.6435,
                0.7689,
                -0.0,
                5.5606,
                -0.7928,
                0.0,
                0.0828,
                0.0238,
                -0.0,
            ],
            [
                0.8167,
                0.7641,
                0.0,
                -0.7928,
                -0.6295,
                -0.0,
                -0.0238,
                -0.1346,
                -0.0,
            ],
            [-0.0, 0.0, 0.8915, 0.0, -0.0, -0.7441, 0.0, -0.0, -0.1474],
            [
                -5.6435,
                -0.7689,
                0.0,
                0.0828,
                -0.0238,
                0.0,
                5.5606,
                0.7928,
                -0.0,
            ],
            [
                -0.8167,
                0.7641,
                0.0,
                0.0238,
                -0.1346,
                -0.0,
                0.7928,
                -0.6295,
                -0.0,
            ],
            [0.0, 0.0, 0.8915, -0.0, -0.0, -0.1474, -0.0, -0.0, -0.7441],
        ]
    )

    # Cannot calculate H^(-1) with a singular Hessian
    with pytest.raises(Exception):
        _ = coords.h_inv

    coords.make_hessian_positive_definite()

    newton_step = -np.dot(coords.h_inv, coords.g)

    # All the steps should be negative in the gradient
    for i in range(9):
        assert np.sign(coords.g[i]) != np.sign(newton_step[i])

    # Step size should be small-ish in all coordinates
    assert np.max(np.abs(newton_step)) < 0.2  # Å


def test_inplace_subtraction():
    coords = CartesianCoordinates(np.array([1.0]))

    coords -= np.array([0.1])
    assert np.isclose(coords[0], 0.9, atol=1e-8)


def test_coords_back_transform_tensor_clear():
    raw_arr = np.arange(6, dtype=float).reshape(2, 3)  # 2 atoms
    cart = CartesianCoordinates(raw_arr)

    dic = DIC.from_cartesian(cart)
    dic.update_g_from_cart_g(np.zeros_like(raw_arr))

    assert dic.g is not None
    assert dic._x.g is not None

    # Updating the positions should clear the gradient array
    dic += np.array([0.001])
    assert dic.g is None and dic._x.g is None


def test_pic_b_no_primitives():
    c = PIC()

    # Cannot calculate a B matrix with no constituent primitive internals
    with pytest.raises(Exception):
        c._calc_B(np.arange(6, dtype=float).reshape(2, 3))


def test_pic_add_sanity_checking():
    c = AnyPIC()
    # pic add should check for primitive type
    c.add(PrimitiveDistance(0, 1))
    with pytest.raises(AssertionError):
        c.add(3)

    # pic append is disallowed
    with pytest.raises(NotImplementedError, match="Please use PIC.add()"):
        c.append(PrimitiveDistance(0, 1))

    # pic should not allow duplicate coordinates to be added
    assert len(c) == 1
    c.add(PrimitiveDistance(1, 0))
    assert len(c) == 1


def test_constrained_distance_satisfied():
    d = ConstrainedPrimitiveDistance(0, 1, value=1.0)

    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

    assert not d.is_satisfied(x)

    x[1, 2] = 1.0
    assert d.is_satisfied(x)


def test_angle_primitive_derivative():
    def numerical_derivative(a, b, h=1e-8):
        y = angle(init_coords)
        coords = init_coords.copy()
        coords[a, int(b)] += h
        y_plus = angle(coords)

        return (y_plus - y) / h

    m = water_mol()
    init_coords = m.coordinates.copy()

    angle = PrimitiveBondAngle(1, 0, 2)
    derivs = angle.derivative(init_coords)
    for atom_idx in (0, 1, 2):
        for component in (0, 1, 2):
            analytic = derivs[3 * atom_idx + component]

            assert np.isclose(
                analytic, numerical_derivative(atom_idx, component), atol=1e-6
            )


def test_angle_primitive_equality():
    assert PrimitiveBondAngle(1, 0, 2) == PrimitiveBondAngle(2, 0, 1)
    assert PrimitiveBondAngle(1, 0, 2) != PrimitiveBondAngle(1, 2, 0)


def test_dihedral_value():
    m = h2o2_mol()
    dihedral = PrimitiveDihedralAngle(2, 0, 1, 3)

    assert np.isclose(
        dihedral(m.coordinates), Angle(100.8, units="deg").to("rad"), atol=1.0
    )


def test_dihedral_primitive_derivative():
    def numerical_derivative(a, b, h=1e-8):
        y = dihedral(init_coords)
        coords = init_coords.copy()
        coords[a, int(b)] += h
        y_plus = dihedral(coords)

        return (y_plus - y) / h

    m = h2o2_mol()
    init_coords = m.coordinates.copy()

    dihedral = PrimitiveDihedralAngle(2, 0, 1, 3)
    analytic = dihedral.derivative(init_coords)
    for atom_idx in (0, 1, 2, 3):
        for k in [0, 1, 2]:
            numerical = numerical_derivative(atom_idx, k)
            assert np.isclose(analytic[3 * atom_idx + k], numerical, atol=1e-6)


def test_dihedral_equality():
    assert PrimitiveDihedralAngle(2, 0, 1, 3) == PrimitiveDihedralAngle(
        2, 0, 1, 3
    )
    assert PrimitiveDihedralAngle(2, 0, 1, 3) == PrimitiveDihedralAngle(
        3, 1, 0, 2
    )


def test_composite_bonds_equality():
    a = CompositeBonds(bonds=[(1, 2), (2, 3)], coeffs=[0.5, 1.2])
    b = CompositeBonds(bonds=[(1, 2), (2, 3)], coeffs=[0.5, 1.2])
    c = CompositeBonds(bonds=[(0, 5), (2, 4)], coeffs=[0.5, 1.2])
    d = CompositeBonds(bonds=[(1, 2), (2, 3)], coeffs=[0.1, 1.2])
    assert a == b
    assert a != c  # different bonds
    assert a != d  # different coefficient


def test_linear_angle():
    acetylene = Molecule(
        atoms=[
            Atom("C", 0.35540, -0.20370, -0.44810),
            Atom("C", -0.37180, 0.21470, 0.40200),
            Atom("H", 1.01560, -0.60550, -1.23530),
            Atom("H", -0.99920, 0.59450, 1.15720),
        ]
    )
    x = CartesianCoordinates(acetylene.coordinates)
    angle = PrimitiveDummyLinearAngle(0, 1, 3, LinearBendType.BEND)
    assert angle._vec_r is None
    _ = angle(x)
    assert angle._vec_r is not None
    old_r_vec = angle._vec_r
    # the dummy atom should not change after the first call
    _ = angle(x + 0.05)
    _ = angle(x - 0.07)
    assert angle._vec_r is old_r_vec

    axis_vec = np.array(np.array(angle._vec_r._data) - x.reshape(-1, 3)[1])
    m_n_vec = acetylene.coordinates[0] - acetylene.coordinates[1]
    assert abs(np.dot(axis_vec, m_n_vec)) < 0.001

    # check that the linear bond complement does not have the same value
    angle2 = PrimitiveDummyLinearAngle(0, 1, 3, LinearBendType.COMPLEMENT)
    assert angle != angle2
    assert not np.isclose(angle(x), angle2(x), rtol=1e-3)

    # for linear angle, swapping the end points changes the definition
    angle3 = PrimitiveDummyLinearAngle(3, 1, 0, LinearBendType.BEND)
    assert angle3 != angle


def test_primitives_consistent_with_mol_values():
    # test that the primitive values are the same as the mol.distance etc.
    h2o2 = h2o2_mol()
    coords = h2o2.coordinates
    dist = PrimitiveDistance(0, 1)
    assert np.isclose(dist(coords), h2o2.distance(0, 1), rtol=1e-8)
    invdist = PrimitiveInverseDistance(1, 2)
    assert np.isclose(invdist(coords), 1 / h2o2.distance(1, 2), rtol=1e-8)
    ang = PrimitiveBondAngle(0, 2, 1)
    assert np.isclose(ang(coords), h2o2.angle(0, 2, 1), rtol=1e-8)
    dihedral = PrimitiveDihedralAngle(2, 0, 1, 3)
    assert np.isclose(dihedral(coords), h2o2.dihedral(2, 0, 1, 3), rtol=1e-8)
    ic = CompositeBonds([(0, 1), (0, 2)], [0.3, 0.7])
    mol_val = 0.3 * h2o2.distance(0, 1) + 0.7 * h2o2.distance(0, 2)
    assert np.isclose(mol_val, ic(coords))


# fmt: off
extra_mols = [
    Molecule(
        atoms=[
            Atom("C", 0.63365, 0.11934, -0.13163),
            Atom("C", -0.63367, -0.11938, 0.13153),
            Atom("H", 1.08517, 1.07993, 0.05600),
            Atom("H", -1.08517, -1.07984, -0.05599),
        ]
    ),
    Molecule(
        atoms=[
            Atom("C", 0.63365, 0.11934, -0.13163),
            Atom("C", -0.63367, -0.11938, 0.13153),
            Atom("H", 1.28230, -0.63391, -0.54779),
            Atom("H", -1.08517, -1.07984, -0.05599),
        ]
    ),  # for testing dihedral derivatives over zero
    Molecule(
        atoms=[
            Atom("C", 0.35540, -0.20370, -0.44810),
            Atom("C", -0.37180, 0.21470, 0.40200),
            Atom("H", 1.01560, -0.60550, -1.23530),
            Atom("H", -0.99920, 0.59450, 1.15720),
        ]
    ),
    feco5_mol(), # for testing linear angles
    h2o2_mol(),
]

test_mols = [
    h2o2_mol(), h2o2_mol(), water_mol(),
    water_mol(), water_mol(), *extra_mols
]
test_prims = [
    PrimitiveDihedralAngle(2, 0, 1, 3), PrimitiveBondAngle(0, 2, 1),
    PrimitiveBondAngle(1, 0, 2), PrimitiveDistance(0, 1),
    PrimitiveInverseDistance(0, 1), PrimitiveDihedralAngle(2, 0, 1, 3),
    PrimitiveDihedralAngle(2, 0, 1, 3),
    PrimitiveDummyLinearAngle(0, 1, 3, LinearBendType.BEND),
    PrimitiveLinearAngle(2, 3, 4, 8, LinearBendType.BEND),
    CompositeBonds([(0, 1), (0, 2)], [1, 1]),
]
# fmt: on


@pytest.mark.parametrize("mol,prim", list(zip(test_mols, test_prims)))
def test_primitive_first_derivs(mol, prim):
    init_coords = CartesianCoordinates(mol.coordinates)
    init_prim = prim(init_coords)

    def numerical_first_deriv(coords, h=1e-8):
        coords = coords.flatten()
        derivs = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            coords[i] += h
            derivs[i] = (prim(coords) - init_prim) / h
            coords[i] -= h
        return derivs

    analytic = prim.derivative(init_coords)
    numeric = numerical_first_deriv(init_coords)
    assert np.allclose(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("mol,prim", list(zip(test_mols, test_prims)))
def test_primitve_second_deriv(mol, prim):
    init_coords = CartesianCoordinates(mol.coordinates)
    init_first_der = prim.derivative(init_coords)

    def numerical_second_deriv(coords, h=1e-8):
        coords = coords.flatten()
        derivs = np.zeros((coords.shape[0], coords.shape[0]))
        for i in range(coords.shape[0]):
            coords[i] += h
            derivs[i] = (prim.derivative(coords) - init_first_der) / h
            coords[i] -= h
        return derivs

    analytic = prim.second_derivative(init_coords)
    # second derivative matrix should be symmetric
    assert np.allclose(analytic, analytic.T)
    numeric = numerical_second_deriv(init_coords)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_repr():
    """Test that each primitive has a representation"""

    prims = [
        PrimitiveInverseDistance(0, 1),
        PrimitiveDistance(0, 1),
        ConstrainedPrimitiveDistance(0, 1, value=1e-3),
        PrimitiveBondAngle(1, 0, 2),
        ConstrainedPrimitiveBondAngle(1, 0, 2, value=1.0),
        PrimitiveDihedralAngle(0, 1, 2, 3),
        PrimitiveLinearAngle(0, 1, 2, 3, LinearBendType.BEND),
        PrimitiveLinearAngle(0, 1, 2, 3, LinearBendType.COMPLEMENT),
        PrimitiveDummyLinearAngle(0, 1, 2, LinearBendType.BEND),
        PrimitiveImproperDihedral(0, 1, 2, 3),
        CompositeBonds(bonds=[(1, 2), (2, 3)], coeffs=[1, 1]),
        ConstrainedCompositeBonds([(1, 2), (2, 3)], [1, 1], 0.2),
    ]

    for p in prims:
        assert repr(p) is not None


def test_dic_large_step_allowed_unconverged_back_transform():
    x = CartesianCoordinates(water_mol().coordinates)
    dic = DIC.from_cartesian(x)

    # unconverged IBT is allowed by default
    dic_unconverged = dic + 1.0 * np.ones(shape=(len(dic),))
    new_x = dic_unconverged.to("cartesian")

    # DIC transform should have moved the cartesian coordinates
    assert not np.allclose(new_x, x)

    # should raise exception if unconverged IBT is disallowed
    dic.allow_unconverged_back_transform = False
    with pytest.raises(CoordinateTransformFailed):
        _ = dic + 1.0 * np.ones(shape=(len(dic),))


def test_constrained_angle_delta():
    q = ConstrainedPrimitiveBondAngle(1, 0, 2, value=np.pi)
    mol = water_mol()
    theta = mol.angle(1, 0, 2)
    x = CartesianCoordinates(mol.coordinates)

    assert np.isclose(q.delta(x), theta - np.pi)


def test_constrained_angle_equality():
    a = ConstrainedPrimitiveBondAngle(1, 0, 2, value=np.pi)
    b = ConstrainedPrimitiveBondAngle(2, 0, 1, value=np.pi)

    assert a == b

    b._theta0 = 0.0
    assert a != b


def test_dics_cannot_be_built_with_incomplete_primitives():
    x = CartesianCoordinates(methane_mol().coordinates)
    primitives = PIC(PrimitiveDistance(0, 1))

    with pytest.raises(RuntimeError):
        _ = DIC.from_cartesian(x=x, primitives=primitives)


def test_pic_generation_linear_angle_ref():
    # Fe(CO)5 with linear Fe-C-O bonds
    m = feco5_mol()
    pic = AnyPIC.from_species(m)

    # check that there are no duplicates
    assert not any(ic1 == ic2 for ic1, ic2 in itertools.combinations(pic, r=2))
    # check that linear bends use reference atoms, not dummy
    assert not any(isinstance(ic, PrimitiveDummyLinearAngle) for ic in pic)
    assert PrimitiveLinearAngle(4, 3, 2, 8, LinearBendType.BEND) in pic
    # for C-Fe-C, only one out-of-plane dihedral should be present
    assert PrimitiveImproperDihedral(3, 5, 2, 1) in pic
    assert sum(isinstance(ic, PrimitiveDihedralAngle) for ic in pic) == 1
    # check degrees of freedom = 3N - 6
    x = m.coordinates.flatten()
    assert np.linalg.matrix_rank(pic.get_B(x)) == 3 * m.n_atoms - 6


def test_pic_generation_linear_angle_dummy():
    # acetylene molecule
    mol = acetylene_mol()
    pic = AnyPIC.from_species(mol)

    # there should not be any usual bond angles
    assert not any(isinstance(ic, PrimitiveBondAngle) for ic in pic)
    # there should not be any linear angles with reference atom
    assert not any(isinstance(ic, PrimitiveLinearAngle) for ic in pic)
    # there should be linear angles with dummy
    assert any(isinstance(ic, PrimitiveDummyLinearAngle) for ic in pic)

    # degrees of freedom = 3N - 5 for linear molecules
    x = mol.coordinates.flatten()
    assert np.linalg.matrix_rank(pic.get_B(x)) == 3 * mol.n_atoms - 5


@work_in_tmp_dir()
def test_pic_generation_disjoint_graph():
    # the algorithm should fully connect the graph
    xyz_string = (
        "16\n\n"
        "C      -0.00247        1.65108        0.05872\n"
        "C       1.19010        1.11169        0.27709\n"
        "C       1.58519       -0.30014        0.31049\n"
        "C       0.05831       -1.54292       -0.45110\n"
        "C      -1.18798       -1.04262        0.13551\n"
        "C      -1.28206        0.99883       -0.23631\n"
        "H      -0.07432        2.73634        0.08639\n"
        "H       2.01755        1.78921        0.47735\n"
        "H       1.70503       -0.70916        1.30550\n"
        "H       2.40398       -0.55376       -0.34855\n"
        "H       0.44229       -2.48695       -0.08638\n"
        "H       0.15289       -1.41865       -1.51944\n"
        "H      -1.25410       -1.13318        1.21833\n"
        "H      -2.09996       -1.35918       -0.36715\n"
        "H      -2.09462        1.29055        0.41495\n"
        "H      -1.56001        1.00183       -1.28217\n"
    )
    with open("diels_alder_complex.xyz", "w") as fh:
        fh.write(xyz_string)

    mol = Molecule("diels_alder_complex.xyz")
    assert not mol.graph.is_connected
    pic = AnyPIC.from_species(mol)

    # shortest bond is between 4, 5 which should also generate angle, torsion
    assert PrimitiveDistance(4, 5) in pic
    assert PrimitiveBondAngle(3, 4, 5) in pic
    assert PrimitiveBondAngle(4, 5, 0) in pic
    assert PrimitiveDihedralAngle(3, 4, 5, 0) in pic

    # the other distance between fragments is 2, 3 which should not be connected
    assert PrimitiveDistance(2, 3) not in pic
    assert PrimitiveBondAngle(1, 2, 3) not in pic
    # check degrees of freedom = 3N - 6
    x = mol.coordinates.flatten()
    assert np.linalg.matrix_rank(pic.get_B(x)) == 3 * mol.n_atoms - 6

    # if the bond between 2, 3 is made into a constraint, it will generate angles
    mol.constraints.distance = {(2, 3): mol.distance(2, 3)}
    pic = AnyPIC.from_species(mol)
    assert ConstrainedPrimitiveDistance(2, 3, mol.distance(2, 3)) in pic
    assert PrimitiveBondAngle(1, 2, 3) in pic


def test_pic_generation_chain_dihedrals():
    # extra dihedrals are needed for ends of linear chains like allene
    cumulene = cumulene_mol()
    pic = AnyPIC.from_species(cumulene)

    assert PrimitiveDihedralAngle(5, 3, 4, 8) in pic
    assert PrimitiveDihedralAngle(6, 3, 4, 7) in pic
    assert PrimitiveDihedralAngle(8, 4, 3, 6) in pic
    assert PrimitiveDihedralAngle(7, 4, 3, 6) in pic

    # check that the 3N-6 degrees of freedom are maintained
    x = cumulene.coordinates.flatten()
    assert np.linalg.matrix_rank(pic.get_B(x)) == 3 * cumulene.n_atoms - 6


def test_pic_generation_square_planar():
    ptcl4 = Molecule(
        atoms=[
            Atom("Pt", -0.1467, -0.2594, -0.0294),
            Atom("Cl", -0.4597, -2.5963, -0.0523),
            Atom("Cl", 2.1804, -0.5689, -0.2496),
            Atom("Cl", -2.4738, 0.0501, 0.1908),
            Atom("Cl", 0.1663, 2.0776, -0.0066),
        ],
        charge=-2,
    )

    # for sq planar, out-of-plane dihedrals are needed to have
    # all degrees of freedom
    pic = AnyPIC.from_species(ptcl4)
    x = ptcl4.coordinates.flatten()
    assert np.linalg.matrix_rank(pic.get_B(x)) == 3 * ptcl4.n_atoms - 6
