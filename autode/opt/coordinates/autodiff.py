"""
Automatic differentiation routines in pure Python
"""
from typing import Union, Callable, Sequence, Optional
from enum import Enum
from copy import deepcopy
import numpy as np
import math

numeric = (float, int)
numeric_type = Union[float, int]


class DerivativeOrder(Enum):
    """Order of derivative"""

    zeroth = 0
    first = 1
    second = 2


def get_differentiable_vars(
    values: Sequence[numeric_type],
    symbols: Sequence[str],
    deriv_order: DerivativeOrder = DerivativeOrder.second,
):
    """
    Obtain differentiable variables from a series of numbers

    Args:
        values: The values of the variables (numbers)
        symbols: List of symbols (strings) of the numbers
        deriv_order: Order of differentiation

    Returns:
        (list[VectorHyperDual]): A list of hyper dual numbers
    """
    assert all(isinstance(sym, str) for sym in symbols)
    assert len(symbols) == len(values)
    symbols = list(symbols)

    hyperduals = []
    for symbol, value in zip(symbols, values):
        var = VectorHyperDual.from_variable(
            value, symbol, all_symbols=symbols, order=deriv_order
        )
        hyperduals.append(var)

    return hyperduals


class VectorHyperDual:
    """
    Hyper-dual numbers with vector infinitesimals upto the
    second order (i.e., upto second partial derivatives)
    """

    def __init__(
        self,
        value: numeric_type,
        symbols: Sequence[str],
        first_der: Optional[np.ndarray] = None,
        second_der: Optional[np.ndarray] = None,
    ):
        """
        Create a vector hyper dual number, i.e. a scalar function
        with one or more variables

        Args:
            value: The scalar value of the hyper-dual
            symbols: A list of unique strings representing the variables
            first_der: 1D array of first derivatives against variables
            second_der: 2D matrix of second derivatives against variables
        """
        assert isinstance(value, numeric)
        self._val = float(value)

        assert all(isinstance(sym, str) for sym in symbols)
        if len(set(symbols)) != len(list(symbols)):
            raise RuntimeError("Symbols must be unique!")
        self._symbols = tuple(symbols)

        # load the derivatives with sanity checks
        self._first_der: Optional[np.ndarray] = None
        self._second_der: Optional[np.ndarray] = None
        self._order = DerivativeOrder.zeroth
        if first_der is None:
            return
        assert isinstance(first_der, np.ndarray)
        first_der = first_der.flatten()
        if not first_der.shape == (self.n_vars,):
            raise ValueError(
                f"Number of symbols ({self.n_vars}) does not match with"
                f" shape of derivative array {first_der.shape}"
            )
        self._first_der = first_der.astype(float)
        self._order = DerivativeOrder.first

        if second_der is None:
            return
        assert isinstance(second_der, np.ndarray)
        if not second_der.shape == (self.n_vars, self.n_vars):
            raise ValueError(
                f"Number of symbols ({self.n_vars}) does not match with"
                f" shape of second derivative matrix {first_der.shape}"
            )
        self._second_der = second_der.astype(float)
        self._order = DerivativeOrder.second

    @property
    def n_vars(self) -> int:
        """Number of variables in this hyper-dual"""
        return len(self._symbols)

    def copy(self):
        return deepcopy(self)

    def _check_compatible(self, other: "VectorHyperDual"):
        if self.n_vars != other.n_vars:
            raise ValueError(
                "Incompatible number of differentiable variables, "
                "cannot perform operation"
            )
        if self._symbols != other._symbols:
            raise ValueError(
                "The differentiable variable symbols do not match, "
                "cannot perform operation!"
            )
        if self._order != other._order:
            raise ValueError("The order of derivative do not match!")
        return None

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, value):
        assert isinstance(value, numeric)
        self._val = float(value)

    @classmethod
    def from_variable(
        cls,
        value: float,
        symbol: str,
        all_symbols: Sequence[str],
        order: DerivativeOrder,
    ):
        """
        Create a hyper-dual number from one variable, requires
        list of symbols and the symbol of this specific variable.
        Essentially a variable x can be considered as a scalar function
        of a list of variables - 1 * x + 0 * y + 0 * z + ...

        Args:
            value: The value of the variable (will be converted to float)
            symbol: The symbol of the current variable, must be in all_symbols
            all_symbols: List of strings indicating all required variables
            order: The order of differentiation to consider

        Returns:
            (VectorHyperDual): The hyper-dual representing the variable
        """
        assert all(isinstance(sym, str) for sym in all_symbols)
        assert isinstance(symbol, str)
        assert symbol in all_symbols

        val = float(value)
        first_der = None
        second_der = None
        n = len(all_symbols)
        idx = list(all_symbols).index(symbol)

        if order == DerivativeOrder.first or order == DerivativeOrder.second:
            first_der = np.zeros(shape=n, dtype=float)
            first_der[idx] = 1.0
        if order == DerivativeOrder.second:
            second_der = np.zeros(shape=(n, n), dtype=float)

        return VectorHyperDual(val, all_symbols, first_der, second_der)

    def differentiate_wrt(
        self,
        symbol1: str,
        symbol2: Union[str, None] = None,
    ) -> Optional[float]:
        """
        Derivative of this hyper-dual number (scalar function) against one
        or two variable(s) identified by their string(s).

        Args:
            symbol1 (str):
            symbol2 (str|None):

        Returns:
            (float|None): The derivative value, or None if not available
        """
        assert isinstance(symbol1, str)
        if symbol1 not in self._symbols:
            return None

        if self._order == DerivativeOrder.zeroth:
            return None

        idx_1 = self._symbols.index(symbol1)
        assert self._first_der is not None
        if symbol2 is None:
            return self._first_der[idx_1]

        assert isinstance(symbol2, str)
        if symbol2 not in self._symbols:
            return None
        idx_2 = self._symbols.index(symbol2)
        # check if second derivs are available
        if self._order == DerivativeOrder.first:
            return None
        assert self._second_der is not None
        return self._second_der[idx_1, idx_2]

    def __add__(
        self, other: Union["VectorHyperDual", numeric_type]
    ) -> "VectorHyperDual":
        """Adding a hyper dual number"""

        # add to a float
        if isinstance(other, numeric):
            new = self.copy()
            new._val += float(other)
            return new

        # add to another dual number
        elif isinstance(other, VectorHyperDual):
            self._check_compatible(other)

            val = self._val + other._val
            if self._order == DerivativeOrder.zeroth:
                return VectorHyperDual(val, self._symbols)

            assert self._first_der is not None
            assert other._first_der is not None
            first_der = self._first_der + other._first_der
            if self._order == DerivativeOrder.first:
                return VectorHyperDual(val, self._symbols, first_der)

            assert self._second_der is not None
            assert other._second_der is not None
            second_der = self._second_der + other._second_der
            return VectorHyperDual(val, self._symbols, first_der, second_der)

        else:
            raise TypeError("Unknown type for addition")

    def __radd__(self, other):
        """Addition is commutative"""
        return self.__add__(other)

    def __neg__(self) -> "VectorHyperDual":
        """Unary negative operation"""
        new = self.copy()
        new._val = -new._val
        if self._order == DerivativeOrder.first:
            new._first_der = -new._first_der
        elif self._order == DerivativeOrder.second:
            new._first_der = -new._first_der
            new._second_der = -new._second_der
        return new

    def __sub__(self, other):
        """Subtraction of hyper dual numbers"""
        return self.__add__(-other)

    def __rsub__(self, other):
        """Reverse subtraction"""
        return other + (-self)

    def __mul__(self, other) -> "VectorHyperDual":
        """Multiply a hyper dual number with float or another hyper dual"""
        if isinstance(other, numeric):
            new = self.copy()
            new._val *= float(other)
            return new

        # multiply with another dual
        elif isinstance(other, VectorHyperDual):
            self._check_compatible(other)

            val = self._val * other._val
            if self._order == DerivativeOrder.zeroth:
                return VectorHyperDual(val, self._symbols)

            assert self._first_der is not None
            assert other._first_der is not None
            first_der = (
                self._val * other._first_der + other._val * self._first_der
            )
            if self._order == DerivativeOrder.first:
                return VectorHyperDual(val, self._symbols, first_der)

            assert self._second_der is not None
            assert other._second_der is not None
            second_der = (
                self._val * other._second_der
                + np.outer(self._first_der, other._first_der)
                + np.outer(other._first_der, self._first_der)
                + other._val * self._second_der
            )
            return VectorHyperDual(val, self._symbols, first_der, second_der)
        else:
            raise TypeError("Unknown type for multiplication")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """True division, defined by multiplicative inverse"""
        return self.__mul__(DifferentiableMath.pow(other, -1))

    def __rtruediv__(self, other):
        """Reverse true division"""
        return DifferentiableMath.pow(self, -1).__mul__(other)

    def __pow__(self, power, modulo=None) -> "VectorHyperDual":
        if modulo is not None:
            raise NotImplementedError("Modulo inverse is not implemented")

        return DifferentiableMath.pow(self, power)

    def __rpow__(self, other):
        return DifferentiableMath.pow(other, self)

    @staticmethod
    def apply_operation(
        num: Union["VectorHyperDual", numeric_type],
        operator: Callable[[float], float],
        operator_first_deriv: Callable[[float], float],
        operator_second_deriv: Callable[[float], float],
    ):
        """
        Perform an operation on the hyperdual (i.e. apply a scalar function)

        Args:
            num: Number that is hyper-dual (or float/int)
            operator: Function that returns the value (result) of operation
            operator_first_deriv: Should return first derivative of operation
            operator_second_deriv: Should return second derivative of operation

        Returns:
            (VectorHyperDual|float): The result
        """
        # pass through numeric types
        if isinstance(num, numeric):
            return operator(float(num))

        val = operator(num._val)

        if num._order == DerivativeOrder.zeroth:
            return VectorHyperDual(val, num._symbols)

        assert num._first_der is not None
        f_dash_x0 = operator_first_deriv(num._val)
        first_der = num._first_der * f_dash_x0

        if num._order == DerivativeOrder.first:
            return VectorHyperDual(val, num._symbols, first_der)

        assert num._second_der is not None
        second_der = np.outer(num._first_der, num._first_der) * (
            operator_second_deriv(num._val)
        )
        second_der += num._second_der * f_dash_x0
        return VectorHyperDual(val, num._symbols, first_der, second_der)


class DifferentiableMath:
    """
    Class defining math functions that can be used on
    hyper dual numbers (i.e. differentiable functions),
    as well as standard numeric types (float and int)
    """

    @staticmethod
    def sqrt(num: Union[VectorHyperDual, numeric_type]):
        """Calculate the square root of a hyperdual number"""

        if isinstance(num, numeric):
            assert num > 0
        else:
            assert num.value > 0

        def value(x0: float):
            return math.sqrt(x0)

        def derivative(x0: float):
            return 1 / (2 * math.sqrt(x0))

        def second_derivative(x0: float):
            return -1 / (4 * math.pow(x0, 3 / 2))

        return VectorHyperDual.apply_operation(
            num, value, derivative, second_derivative
        )

    @staticmethod
    def exp(num: Union[VectorHyperDual, numeric_type]):
        """Raise e to the power of num"""

        def value(x0: float):
            return math.exp(x0)

        def derivative(x0: float):
            return math.exp(x0)

        def second_derivative(x0: float):
            return math.exp(x0)

        return VectorHyperDual.apply_operation(
            num, value, derivative, second_derivative
        )

    @staticmethod
    def pow(
        num: Union[VectorHyperDual, numeric_type],
        power: Union[VectorHyperDual, numeric_type],
    ):
        """Exponentiation"""

        if isinstance(num, numeric) and isinstance(power, numeric):
            return math.pow(num, power)

        elif isinstance(num, VectorHyperDual) and isinstance(power, numeric):
            if num.value < 0 and isinstance(power, float):
                raise ValueError(
                    "Math error, can't raise negative number to fractional power"
                )

            def value(x0: float):
                return math.pow(x0, power)

            def derivative(x0: float):
                return power * math.pow(x0, power - 1)

            def second_derivative(x0: float):
                return power * (power - 1) * math.pow(x0, power - 2)

            return VectorHyperDual.apply_operation(
                num, value, derivative, second_derivative
            )

        elif isinstance(power, VectorHyperDual):
            if (isinstance(num, numeric) and num) < 0 or (
                isinstance(num, VectorHyperDual) and num.value < 0
            ):
                raise ValueError(
                    "Only positive numbers can be used with"
                    " differentiable exponent"
                )
            # use identity x^y = e^(y log_x) for x > 0
            return DifferentiableMath.exp(power * DifferentiableMath.log(num))

    @staticmethod
    def log(num: Union[VectorHyperDual, numeric_type]):
        """Natural logarithm"""

        if isinstance(num, numeric):
            assert num > 0
        else:
            assert num.value > 0

        def value(x0: float):
            return math.log(x0)

        def derivative(x0: float):
            return 1.0 / x0

        def second_derivative(x0: float):
            return -1.0 / (x0**2)

        return VectorHyperDual.apply_operation(
            num, value, derivative, second_derivative
        )

    @staticmethod
    def acos(num: Union[VectorHyperDual, numeric_type]):
        """Calculate the arccosine of a hyperdual number"""

        if isinstance(num, VectorHyperDual):
            assert -1 < num.value < 1
        else:
            assert -1 < num < 1

        def value(x0: float):
            return math.acos(x0)

        def derivative(x0: float):
            return -1 / math.sqrt(1 - x0**2)

        def second_derivative(x0: float):
            return -x0 / math.pow(1 - x0**2, 3 / 2)

        return VectorHyperDual.apply_operation(
            num, value, derivative, second_derivative
        )

    @staticmethod
    def atan(num: Union[VectorHyperDual, numeric_type]):
        """Calculate the arctangent of a hyperdual number"""

        def value(x0: float):
            return math.atan(x0)

        def derivative(x0: float):
            return 1 / (1 + x0**2)

        def second_derivative(x0: float):
            return (-2 * x0) / (x0**2 + 1) ** 2

        return VectorHyperDual.apply_operation(
            num, value, derivative, second_derivative
        )

    @staticmethod
    def atan2(
        num_y: Union[VectorHyperDual, numeric_type],
        num_x: Union[VectorHyperDual, numeric_type],
    ):
        """Calculate the arctan2 of two hyper dual numbers"""
        if isinstance(num_y, numeric) and isinstance(num_x, numeric):
            return math.atan2(num_y, num_x)

        def atan2_derivs_x_not_0(y, x):
            return DifferentiableMath.atan(y / x)

        def atan2_derivs_x_close_0(y, x):
            return -DifferentiableMath.atan(x / y)

        x_val = float(num_x) if isinstance(num_x, numeric) else num_x.value
        y_val = float(num_y) if isinstance(num_y, numeric) else num_y.value
        res_val = math.atan2(y_val, x_val)

        # when atan2(y,x)->pi/2, x->0 or y/x->inf, use other formula for derivs
        if math.isclose(abs(res_val), math.pi / 2, abs_tol=0.1):
            res = atan2_derivs_x_close_0(num_y, num_x)
            res.value = res_val
            return res
        else:
            res = atan2_derivs_x_not_0(num_y, num_x)
            res.value = res_val
            return res
