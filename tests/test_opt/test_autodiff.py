import pytest
from autode.opt.coordinates.autodiff import (
    get_differentiable_vars,
    DifferentiableMath,
)


def test_hyperdual_sanity_checks():
    a, b = get_differentiable_vars(
        values=[1, 2], symbols=["a", "b"], deriv_order=1
    )

    c = get_differentiable_vars(values=[1], symbols=["c"], deriv_order=1)[0]
    with pytest.raises(ValueError, match="Incompatible number"):
        _ = a + c

    c, d = get_differentiable_vars(
        values=[1, 2], symbols=["c", "d"], deriv_order=1
    )
    with pytest.raises(ValueError, match="symbols do not match"):
        _ = a * c

    a2, b2 = get_differentiable_vars(
        values=[1, 2], symbols=["a", "b"], deriv_order=2
    )
    with pytest.raises(ValueError, match="order of derivative do not match!"):
        _ = DifferentiableMath.atan2(a2, b)

    with pytest.raises(RuntimeError, match="Symbols must be unique"):
        _ = get_differentiable_vars(
            values=[1, 2], symbols=["a", "a"], deriv_order=1
        )
