import math
import numpy as np
import pytest
from autode.opt.coordinates._autodiff import (
    get_differentiable_vars,
    DifferentiableMath,
    VectorHyperDual,
    DerivativeOrder,
    DifferentiableVector3D,
)


def test_hyperdual_sanity_checks():
    a, b = get_differentiable_vars(
        values=[1, 2], symbols=["a", "b"], deriv_order=DerivativeOrder.first
    )
    assert repr(a) is not None

    c = get_differentiable_vars(
        values=[1], symbols=["c"], deriv_order=DerivativeOrder.first
    )[0]
    with pytest.raises(ValueError, match="Incompatible number"):
        _ = a + c

    c, d = get_differentiable_vars(
        values=[1, 2], symbols=["c", "d"], deriv_order=DerivativeOrder.first
    )
    with pytest.raises(ValueError, match="symbols do not match"):
        _ = a * c

    a2, b2 = get_differentiable_vars(
        values=[1, 2], symbols=["a", "b"], deriv_order=DerivativeOrder.second
    )
    with pytest.raises(ValueError, match="order of derivative do not match!"):
        _ = DifferentiableMath.atan2(a2, b)

    with pytest.raises(RuntimeError, match="Symbols must be unique"):
        _ = get_differentiable_vars(
            values=[1, 2],
            symbols=["a", "a"],
            deriv_order=DerivativeOrder.first,
        )


def x_to_the_y_derivs(x, y):
    d_dx = y * x ** (y - 1)
    d_dy = x**y * math.log(x)
    return d_dx, d_dy


def x_to_the_y_second_derivs(x, y):
    d2_dx2 = y * (y - 1) * x ** (y - 2)
    d2_dxdy = math.log(x) * y * x ** (y - 1) + x**y * (1 / x)
    d2_dydx = x ** (y - 1) + y * x ** (y - 1) * math.log(x)
    d2_dy2 = math.log(x) * math.log(x) * x**y
    return d2_dx2, d2_dxdy, d2_dydx, d2_dy2


def test_autodiff_exponential_func():
    x_val, y_val = 1.0, 2.0
    x, y = get_differentiable_vars(
        values=[x_val, y_val],
        symbols=["x", "y"],
    )
    result = x**y
    d_dx, d_dy = x_to_the_y_derivs(x_val, y_val)
    assert math.isclose(result.differentiate_wrt("x"), d_dx)
    assert math.isclose(result.differentiate_wrt("y"), d_dy)

    d2_dx2, d2_dxdy, d2_dydx, d2_dy2 = x_to_the_y_second_derivs(x_val, y_val)
    assert math.isclose(result.differentiate_wrt("x", "x"), d2_dx2)
    assert math.isclose(result.differentiate_wrt("x", "y"), d2_dxdy)
    assert math.isclose(result.differentiate_wrt("y", "x"), d2_dydx)
    assert math.isclose(result.differentiate_wrt("y", "y"), d2_dy2)
    # check the second derivatives are symmetric
    assert math.isclose(d2_dydx, d2_dxdy)


def test_exponential_math_sanity_checks():
    x, y = get_differentiable_vars([-0.1, -0.2], ["x", "y"])
    assert x**2 is not None
    assert x**-1 is not None
    # negative number raised to fractional power is complex
    with pytest.raises(AssertionError):
        _ = x**0.2
    # negative number cannot be raised to differentiable power
    with pytest.raises(AssertionError):
        _ = (-2) ** y
    with pytest.raises(AssertionError):
        _ = x**y


def test_math_funcs_work_with_native_types():
    # python float or int types should be passed through
    assert math.isclose(DifferentiableMath.acos(0), math.pi / 2)
    (y,) = get_differentiable_vars([1], ["y"])
    res = DifferentiableMath.atan2(y, 0)
    assert isinstance(res, VectorHyperDual)
    assert math.isclose(res.value, math.pi / 2)
    res = DifferentiableMath.atan2(1, 0)
    assert isinstance(res, float)
    assert math.isclose(DifferentiableMath.pow(0.9, 1.3), math.pow(0.9, 1.3))

    # however, python's complex type is not supported
    with pytest.raises(TypeError, match="Unknown type for addition"):
        _ = y + (4 + 1j)
    with pytest.raises(TypeError, match="Unknown type for multiplication"):
        _ = y * (1 + 2j)
    with pytest.raises(TypeError, match="Unknown type for exponentiation"):
        _ = y ** (1 + 2j)


def test_hyperdual_init_checks():
    with pytest.raises(ValueError):
        _ = VectorHyperDual(
            0.1, symbols=["x", "y"], first_der=np.array([0.1, 0.2, 0.3])
        )

    with pytest.raises(ValueError):
        _ = VectorHyperDual(
            0.1,
            symbols=["x", "y"],
            first_der=np.array([0.1, 0.2]),
            second_der=np.zeros(shape=(2, 3)),
        )


def test_hyperdual_order():
    symbols = ["x", "y"]
    x = VectorHyperDual(
        0.1,
        symbols=symbols,
    )
    assert x._order == DerivativeOrder.zeroth
    x = VectorHyperDual(0.1, symbols=symbols, first_der=np.zeros(2))
    assert x._order == DerivativeOrder.first
    # will ignore second derivatives if first is not present
    x = VectorHyperDual(0.1, symbols=symbols, second_der=np.zeros((3, 2)))
    assert x._order == DerivativeOrder.zeroth
    x = VectorHyperDual(
        0.1,
        symbols=symbols,
        first_der=np.zeros(2),
        second_der=np.zeros((2, 2)),
    )
    assert x._order == DerivativeOrder.second


def test_derivative_not_available():
    x, y = get_differentiable_vars(
        [1.0, 2.0], symbols=["x", "y"], deriv_order=DerivativeOrder.first
    )
    res = 1 - x**2 + y
    assert isinstance(res, VectorHyperDual)
    assert math.isclose(res.value, 2)
    assert math.isclose(res.differentiate_wrt("x"), -2)
    # higher order derivatives are not available
    assert res.differentiate_wrt("x", "y") is None
    # unknown variables
    assert res.differentiate_wrt("z") is None
    assert res.differentiate_wrt("y", "z") is None

    # with zero order, only value is present
    (x,) = get_differentiable_vars(
        [1.0], symbols=["x"], deriv_order=DerivativeOrder.zeroth
    )
    res = 1 + x**2
    assert isinstance(res, VectorHyperDual)
    assert res.differentiate_wrt("x") is None


def test_hyperdual_3d_vector_sanity_checks():
    x, y, z = get_differentiable_vars(
        [0.1, 0.2, 0.3],
        ["x", "y", "z"],
    )

    with pytest.raises(ValueError):
        _ = DifferentiableVector3D([x, y])

    vec1 = DifferentiableVector3D([x, y, z])
    vec2 = DifferentiableVector3D([x, y, z])

    with pytest.raises(ValueError):
        _ = vec1.dot(2)

    with pytest.raises(AssertionError):
        _ = vec1 * vec2

    assert 2 * vec1 is not None
