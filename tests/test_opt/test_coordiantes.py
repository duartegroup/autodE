import pytest
import numpy as np
from autode import Molecule, Atom
from autode.opt.coordinates.internals import InverseDistances
from autode.opt.coordinates.primitives import InverseDistance, Distance
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.opt.coordinates.dic import DIC


def methane_mol():
    return Molecule(atoms=[Atom('C',  0.11105, -0.21307,  0.00000),
                           Atom('H',  1.18105, -0.21307,  0.00000),
                           Atom('H', -0.24562, -0.89375,  0.74456),
                           Atom('H', -0.24562, -0.51754, -0.96176),
                           Atom('H', -0.24562,  0.77207,  0.21720)])


def h2():
    return Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.5)])


def test_inv_dist_primitives():

    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    inv_dist = InverseDistance(0, 1)
    assert np.isclose(inv_dist(x), 0.5)     # 1/2.0 = 0.5 Å-1

    # Check a couple of derivatives by hand
    assert np.isclose(inv_dist.derivative(0, 'x', x=x),
                      2*inv_dist(x)**3)
    assert np.isclose(inv_dist.derivative(1, 'x', x=x),
                      -2*inv_dist(x)**3)

    # Derivatives with respect to zero components
    assert np.isclose(inv_dist.derivative(0, 'y', x=x),
                      0)
    # or those that are not present in the system should be zero
    assert np.isclose(inv_dist.derivative(2, 'x', x=x),
                      0)


def test_dist_primitives():

    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    inv_dist = Distance(0, 1)
    assert np.isclose(inv_dist(x), 2.0)

    assert np.isclose(inv_dist.derivative(0, 'x', x=x),
                      -2/2)

    assert np.isclose(inv_dist.derivative(1, 'x', x=x),
                      +2/2)

    for component in ('y', 'z'):
        assert np.isclose(inv_dist.derivative(1, component, x=x),
                          0)

    assert np.isclose(inv_dist.derivative(2, 'x', x=x),
                      0)


def test_primitive_equality():

    assert InverseDistance(0, 1) != 'a'
    assert InverseDistance(0, 1) == InverseDistance(0, 1)
    assert InverseDistance(1, 0) == InverseDistance(0, 1)


def test_primitives_equality():

    x = CartesianCoordinates(h2().coordinates)
    primitives = InverseDistances.from_cartesian(x)

    assert primitives != 'a'
    assert primitives == InverseDistances.from_cartesian(x)
    
    # Order does not matter for equality
    assert primitives == InverseDistances(InverseDistance(1, 0))


def test_cartesian_coordinates():
    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)  # defaults to Å
    assert 'cart' in repr(x).lower()
    assert x.ndim == 1

    # Test equality
    assert x != 'some string'
    assert x != arr

    # Can convert to other distance units
    assert 0.19 < CartesianCoordinates(arr).to('nm')[3] < 0.21

    # but not an unsupported unit
    with pytest.raises(ValueError):
        _ = CartesianCoordinates(arr).to('X')


def test_cartesian_coordinates_hessian_update():

    # Simple coordinates with 2 atoms in 3 D
    coords = CartesianCoordinates(np.arange(0, 6).reshape((2, 3)))

    with pytest.raises(ValueError):
        coords.update_h_from_cart_h(arr=np.array([]))

    with pytest.raises(ValueError):
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
    for invalid_h in (3, np.array([1.0]), np.arange(3),
                      np.arange(9).reshape(3, 3), np.arange(6).reshape(2, 3)):

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
    assert np.allclose(coords.h_inv, expected_h_inv, atol=1E-10)


def test_cartesian_update_clear():

    arr = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])

    # Gradients and Hessians are initialized to None
    x = CartesianCoordinates(arr)
    assert x.g is None and x.h is None

    x.g = np.array([[0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0]])
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
    assert 'dic' in repr(x).lower()

    with pytest.raises(Exception):
        _ = x.to('unknown coordinates')


def test_invalid_pic_construction():

    # Cannot construct some primitives e.g. InverseDistances from non Primitive
    # internal coordinates
    with pytest.raises(ValueError):
        _ = InverseDistances('a')


def test_cart_to_dic():

    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(arr)

    # Should only have 1 internal coordinate
    pic = InverseDistances.from_cartesian(x)
    assert len(pic) == 1

    # and not have a B matrix
    with pytest.raises(AttributeError):
        _  = pic.B

    # Delocalised internals should preserve the single internal coordinate
    dics = x.to('dic')
    assert isinstance(dics, DIC)
    assert len(dics) == 1
    # and store the previous catesian coordinates
    assert hasattr(dics, '_x')
    # as a copy, so changing the initial x should not change prev_x
    assert id(dics._x) != id(x)

    x += 0.1
    assert np.allclose(x,  np.array([0.1, 0.1, 0.1, 2.1, 0.1, 0.1]))
    assert not np.allclose(x, dics._x)
    x -= 0.1


def test_simple_dic_to_cart():
    arr = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]])

    # Should be able to transform back to Cartesians
    dic = CartesianCoordinates(arr).to('dic')

    x = dic.to('cartesian')
    assert np.allclose(CartesianCoordinates(arr), x)

    assert np.isclose(0.6, (dic + 0.1)[0], atol=1E-6)
    # Updating the DICs should afford cartesian coordinates that are
    # ~1.7 Å apart (1/r = 0.6)
    dic.update(delta=0.1)
    assert dic.shape == (1,)
    assert np.isclose(dic[0], 0.6)

    arr_update = dic.to('cart').reshape((2, 3))

    assert np.isclose(np.linalg.norm(arr_update[0, :] - arr_update[1, :]),
                      1.66666,
                      atol=1E-4)


def test_methane_cart_to_dic():

    x = CartesianCoordinates(methane_mol().coordinates)
    dic = x.to('dic')
    assert len(dic) == 9   # 3N-6 for N=5

    dic.update(delta=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Cartesian coordinates should be close to the starting ones
    assert np.linalg.norm(x - dic.to('cart')) < 0.5


def test_co2_cart_to_dic():

    arr = np.array([[-1.31254, 0.34625, -0.00000],
                    [-0.11672, 0.30964, 0.00000],
                    [1.07904, 0.27311, 0.00000]])

    x = CartesianCoordinates(arr)
    dic = x.to('dic')
    assert len(dic) == 3

    # Applying a shift to the internal coordinates that are close to linear
    # can break the back transformation to Cartesians
    with pytest.raises(RuntimeError):
        dic.update(delta=np.array([0.0, 0.0, 0.1]))


def test_grad_transform_linear():

    k = 1.0
    r0 = 1.0

    def energy(_x):
        """Harmonic potential: E = k(r-r0)^2"""
        _x = _x.reshape((-1, 3))
        r = np.linalg.norm(_x[0] - _x[1])
        return 0.5 * k * (r - r0)**2

    def grad(_x):
        _x = _x.reshape((-1, 3))
        diff = _x[0, 0] - _x[1, 0]
        r = np.linalg.norm(_x[0] - _x[1])
        return np.array([k * (r - r0) * diff/r,
                         0.0,
                         0.0,
                         - k * (r - r0) * diff/r,
                         0.0,
                         0.0])

    def num_grad(_x, h=1E-8):

        _g = []
        for i in range(len(_x.flatten())):

            x_ph = np.array(_x, copy=True)
            x_ph[i] += h

            g_i = (energy(x_ph) - energy(_x)) / h
            _g.append(g_i)

        return np.array(_g)

    coords = np.array([[0.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0]])

    x = CartesianCoordinates(coords)

    assert np.allclose(num_grad(x), grad(x))
    x.g = grad(coords)

    dic = x.to('dic')
    assert dic.shape == (1,)                   # Only a single distance
    assert np.isclose(dic[0], 0.5, atol=1E-6)  # 1/r_012 = 0.5 Å

    assert dic.g.shape == (1,)   # dE/ds_i has only a single component

    # Determined by hand
    assert np.isclose(dic.g[0],
                      -1/0.5**2 * grad(coords)[3])


def test_hess_transform_linear():

    k = 1.0
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]])

    x = CartesianCoordinates(coords)

    def energy(_x, r0=1):
        """Harmonic potential: E = k(r-r0)^2"""
        _x = _x.reshape((-1, 3))
        r = np.linalg.norm(_x[0] - _x[1])
        return 0.5 * k * (r - r0)**2

    def hessian(_x):
        _x = _x.reshape((-1, 3))
        delta_x = _x[0, 0] - _x[1, 0]

        r = np.linalg.norm(_x[0] - _x[1])
        h_oo = k*(1 - 1/r + delta_x**2/r**3)

        _h = np.zeros(shape=(6, 6))
        _h[0, 0] = _h[3, 3] = h_oo
        _h[0, 3] = _h[3, 0] = -h_oo

        return _h

    def g_i(_x, i, h=1E-8):
        """Numerical graident"""
        x_ph = np.array(_x, copy=True)
        x_ph[i] += h
        return (energy(x_ph) - energy(_x)) / h

    def h_ij(_x, i, j, h=1E-8):

        x_ph = np.array(_x, copy=True)
        x_ph[j] += h

        return (g_i(x_ph, i=i) - g_i(_x, i=i)) / h

    def num_hess(_x):
        _h = np.zeros(shape=(6, 6))

        for i in range(6):
            for j in range(6):

                _h[i, j] = h_ij(_x, i, j)

        return _h

    assert np.linalg.norm(hessian(x) - num_hess(x)) < 1E-7
    x.h = hessian(coords)

    dic = x.to('dic')
    assert dic.h.shape == (1, 1)         # 1x1 internal Hessian
    assert np.isclose(dic.h[0, 0], k)    # should be ~k

