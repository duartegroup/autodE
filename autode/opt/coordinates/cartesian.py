import numpy as np
from typing import Optional, TYPE_CHECKING

from autode.log import logger
from autode.values import ValueArray
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.dic import DIC

if TYPE_CHECKING:
    from autode.values import Gradient
    from autode.hessians import Hessian


class CartesianCoordinates(OptCoordinates):
    """Flat Cartesian coordinates shape = (3 × n_atoms, )"""

    def __repr__(self):
        return f"Cartesian Coordinates({np.ndarray.__str__(self)} {self.units.name})"

    def __new__(cls, input_array, units="Å") -> "CartesianCoordinates":
        """New instance of these coordinates"""

        # if it has units cast into current units
        if isinstance(input_array, ValueArray):
            input_array = ValueArray.to(input_array, units=units)

        return super().__new__(
            cls, np.array(input_array).flatten(), units=units
        )

    def __array_finalize__(self, obj) -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        return None if obj is None else super().__array_finalize__(obj)

    def _str_is_valid_unit(self, string) -> bool:
        """Is a string a valid unit for these coordinates e.g. nm"""
        return any(string in unit.aliases for unit in self.implemented_units)

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient, which for
        Cartesian coordinates there is nothing to be done for.

        -----------------------------------------------------------------------
        Arguments:
            arr: Gradient array
        """
        self.g = None if arr is None else np.array(arr).flatten()

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the Hessian from a Cartesian Hessian matrix with shape
        3N x 3N for a species with N atoms.


        -----------------------------------------------------------------------
        Arguments:
            arr: Hessian matrix
        """
        self.h = None if arr is None else np.array(arr)

    def iadd(self, value: np.ndarray) -> OptCoordinates:
        return np.ndarray.__iadd__(self, value)

    def to(self, value: str) -> OptCoordinates:
        """
        Transform between cartesian and internal coordinates e.g. delocalised
        internal coordinates or other units

        -----------------------------------------------------------------------
        Arguments:
            value (str): Intended conversion

        Returns:
            (autode.opt.coordinates.OptCoordinates): Transformed coordinates

        Raises:
            (ValueError): If the conversion cannot be performed
        """
        logger.info(f"Transforming Cartesian coordinates to {value}")

        if value.lower() in ("cart", "cartesian", "cartesiancoordinates"):
            return self

        elif value.lower() in ("dic", "delocalised internal coordinates"):
            return DIC.from_cartesian(self)

        # ---------- Implement other internal transformations here -----------

        elif self._str_is_valid_unit(value):
            return CartesianCoordinates(
                ValueArray.to(self, units=value), units=value
            )
        else:
            raise ValueError(
                f"Cannot convert Cartesian coordinates to {value}"
            )

    @property
    def expected_number_of_dof(self) -> int:
        """Expected number of degrees of freedom for the system"""
        n_atoms = len(self.flatten()) // 3
        return 3 * n_atoms - 6


class CartesianWithConstraints(CartesianCoordinates):
    def __new__(cls, input_array, units="Å") -> "CartesianWithConstraints":
        """New instance of these coordinates"""

        # if it has units cast into current units
        if isinstance(input_array, ValueArray):
            input_array = ValueArray.to(input_array, units=units)

        arr = super().__new__(
            cls, np.array(input_array).flatten(), units=units
        )
        arr._lambda = None  # lagrangian multipliers
        arr._constraints = None  # constraint functions
        return arr

    def __array_finalize__(self, obj) -> None:
        super().__array_finalize__(obj)
        for attr in ["_lambda", "_constraints"]:
            setattr(self, attr, getattr(obj, attr, None))

    def set_constraints(self, constraints) -> None:
        """
        Add constraint functions, in the form of constrained primitives

        Args:
            constraints:
        """
        from autode.opt.coordinates.primitives import ConstrainedPrimitive

        assert all(
            isinstance(constr, ConstrainedPrimitive) for constr in constraints
        )
        self._constraints = list(constraints)

    @property
    def g(self):
        """Gradient of the energy, with constraint terms"""
        # jacobian of the constraints
        A_mat = np.zeros(shape=(self.shape[0], len(self._constraints)))
        for idx, constr in enumerate(self._constraints):
            A_mat[:, idx] = constr.derivative(np.array(self))
        g_constr = self._g - np.matmul(A_mat, self._lambda).flatten()

        c_vals = [
            -constr.delta(np.array(self)) for constr in self._constraints
        ]
        dL_dx = np.array(g_constr.tolist() + c_vals)
        return dL_dx

    @g.setter
    def g(self, value):
        """Setting a gradient is not allowed with constraints"""
        raise RuntimeError("Cannot set gradients since constraints are added")
