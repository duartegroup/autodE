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
