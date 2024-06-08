import numpy as np
from typing import Optional, List, TYPE_CHECKING

from autode.log import logger
from autode.values import ValueArray
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.dic import DIC

if TYPE_CHECKING:
    from autode.values import Gradient
    from autode.hessians import Hessian
    from autode.opt.coordinates.primitives import ConstrainedPrimitive


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
        self._g = None if arr is None else np.array(arr).flatten()

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the Hessian from a Cartesian Hessian matrix with shape
        3N x 3N for a species with N atoms.


        -----------------------------------------------------------------------
        Arguments:
            arr: Hessian matrix
        """
        self._h = None if arr is None else np.array(arr)

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
    """Cartesian coordinates, with added constraints"""

    def __new__(cls, input_array, units="Å") -> "CartesianWithConstraints":
        """New instance of these coordinates"""

        # if it has units cast into current units
        if isinstance(input_array, ValueArray):
            input_array = ValueArray.to(input_array, units=units)

        arr = super().__new__(
            cls, np.array(input_array).flatten(), units=units
        )
        arr._lambda = np.array([])  # lagrangian multipliers
        arr._constraints = []  # constraint functions
        return arr

    def __array_finalize__(self, obj) -> None:
        super().__array_finalize__(obj)
        for attr in ["_lambda", "_constraints"]:
            setattr(self, attr, getattr(obj, attr, None))

    @property
    def raw(self) -> np.ndarray:
        """Raw numpy array of these coordinates including the multipliers"""
        return np.array(self.tolist() + self._lambda.tolist(), copy=True)

    @property
    def n_constraints(self):
        """Number of constraints"""
        return len(self._constraints)

    @property
    def active_indexes(self) -> List[int]:
        """For Cartesian no specific coordinate can be inactive"""
        return list(range(len(self) + self.n_constraints))

    @property
    def inactive_indexes(self) -> List[int]:
        """No coordinate is inactive"""
        return []

    @property
    def n_satisfied_constraints(self) -> int:
        """How many constraints are satisfied?"""
        return sum(c.is_satisfied(self) for c in self._constraints)

    @classmethod
    def from_cartesian(
        cls,
        x: CartesianCoordinates,
        constraints: List["ConstrainedPrimitive"],
    ):
        """
        Create Cartesian coordinates with optional constraints enforced
        with Lagrange multipliers

        Args:
            x: The original Cartesian coordinates, without any constraints

            constraints: List of constraint primitives (i.e.
                        bonds, angles etc.)

        Returns:
            (CartesianWithConstraints):
        """
        constraints = [] if constraints is None else constraints
        from autode.opt.coordinates.primitives import ConstrainedPrimitive

        assert all(isinstance(c, ConstrainedPrimitive) for c in constraints)

        x_c = cls(x)
        if len(constraints) > 0:
            x_c._constraints = list(constraints)
            x_c._lambda = np.zeros(shape=(len(constraints),))
        return x_c

    def iadd(self, value: np.ndarray) -> OptCoordinates:
        """Addition of an array to this set of coordinates"""
        assert isinstance(value, np.ndarray)
        assert len(value) == len(self) + self.n_constraints

        # separate the coordinates and multipliers
        if self.n_constraints > 0:
            delta_lambda = value[-self.n_constraints :]
            self._lambda += delta_lambda
            delta_x = value[: -self.n_constraints]
        else:
            delta_x = value
        return super().iadd(delta_x)

    @property
    def g(self):
        """Gradient of the energy, with constraint terms"""
        # jacobian of the constraints
        a_mat = np.zeros(shape=(self.shape[0], len(self._constraints)))
        for idx, c in enumerate(self._constraints):
            a_mat[:, idx] = c.derivative(np.array(self))
        g_constr = self._g - np.matmul(a_mat, self._lambda).flatten()
        c_vals = [-c.delta(np.array(self)) for c in self._constraints]
        return np.append(g_constr, c_vals)

    @g.setter
    def g(self, value):
        """Setting a gradient is not allowed with constraints"""
        raise RuntimeError("Cannot set gradients since constraints are added")

    @property
    def h(self):
        """Hessian of the energy, with constraint terms"""
        n, m = self.shape[0], len(self._constraints)
        a_mat = np.zeros(shape=(n, m))
        w_mat = np.array(self._h)
        for idx, c in enumerate(self._constraints):
            a_mat[:, idx] = c.derivative(np.array(self))
            w_mat -= self._lambda[idx] * c.second_derivative(np.array(self))
        # Hessian of the Lagrangian
        d2L_dx2 = np.zeros(shape=(n + m, n + m))
        d2L_dx2[:n, :n] = w_mat
        d2L_dx2[n:, :n] = a_mat.T
        d2L_dx2[:n, n:] = a_mat
        return d2L_dx2

    @h.setter
    def h(self, value):
        """Setting a hessian is not allowed with constraints"""
        raise RuntimeError("Cannot set Hessian since constraints are added")

    def to(self, value: str) -> OptCoordinates:
        """Convert to other coordinate types"""
        if value.lower() in ("cart", "cartesian", "cartesiancoordinates"):
            coords = CartesianCoordinates(np.array(self))
            coords.update_g_from_cart_g(self._g)
            coords.update_h_from_cart_h(self._h)
            return coords

        elif value.lower() in ("dic", "delocalised internal coordinates"):
            return self.to("cart").to("dic")

        else:
            raise ValueError(f"Cannot convert to {value}")
