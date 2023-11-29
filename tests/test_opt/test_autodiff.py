import math

import pytest
from autode.opt.coordinates.autodiff import (
    get_differentiable_vars,
    DifferentiableMath,
    VectorHyperDual,
    DerivativeOrder,
)


def test_hyperdual_sanity_checks():
    a, b = get_differentiable_vars(
        values=[1, 2], symbols=["a", "b"], deriv_order=DerivativeOrder.first
    )

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


def x_to_the_y(x, y):
    return x**y


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
