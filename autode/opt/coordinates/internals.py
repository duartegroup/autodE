"""
Internal coordinates. Notation follows:


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
G : Spectroscopic G matrix
"""
import numpy as np

from typing import Any, Optional, Type, List
from abc import ABC, abstractmethod
from autode.opt.coordinates.base import OptCoordinates, CartesianComponent
from autode.opt.coordinates.primitives import (
    InverseDistance,
    Primitive,
    Distance,
    DihedralAngle,
)


class InternalCoordinates(OptCoordinates, ABC):  # lgtm [py/missing-equals]
    def __new__(cls, input_array) -> "InternalCoordinates":
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array, units=None)

        for attr in ("_x", "primitives"):
            setattr(arr, attr, getattr(input_array, attr, None))

        return arr

    def __array_finalize__(self, obj: "OptCoordinates") -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        OptCoordinates.__array_finalize__(self, obj)

        for attr in ("_x", "primitives"):
            setattr(self, attr, getattr(obj, attr, None))

        return

    @property
    def n_constraints(self) -> int:
        """Number of constraints in these coordinates"""
        return self.primitives.n_constrained

    @property
    def constrained_primitives(self) -> List["ConstrainedPrimitive"]:
        return [p for p in self.primitives if p.is_constrained]

    @property
    def n_satisfied_constraints(self) -> int:
        """Number of constraints that are satisfied in these coordinates"""
        x = self.to("cartesian")
        return sum(p.is_satisfied(x) for p in self.constrained_primitives)


class PIC(list, ABC):
    """Primitive internal coordinates"""

    def __init__(self, *args):
        """
        List of primitive internal coordinates with a Wilson B matrix.
        If there are no arguments then all possible primitive coordinates
        will be generated
        """
        super().__init__(args)

        self._B: Optional[np.ndarray] = None

        if not self._are_all_primitive_coordinates(args):
            raise ValueError(
                "Cannot construct primitive internal coordinates "
                f"from {args}. Must be primitive internals"
            )

    @property
    def B(self) -> np.ndarray:
        """Wilson B matrix"""

        if self._B is None:
            raise AttributeError(
                f"{self} had no B matrix. Please calculate "
                f"the value of the primitives to determine B"
            )

        return self._B

    @property
    def G(self) -> np.ndarray:
        """Spectroscopic G matrix as the symmetrised Wilson B matrix"""
        return np.dot(self.B, self.B.T)

    @classmethod
    def from_cartesian(
        cls,
        x: "autode.opt.cartesian.CartesianCoordinates",
    ) -> "PIC":
        """Construct a complete set of primitive internal coordinates from
        a set of Cartesian coordinates"""

        pic = cls()
        pic._populate_all(x=x)

        return pic

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Populate Primitive-s used in the construction of set"""

        q = self._calc_q(x)
        self._calc_B(x)

        return q

    def close_to(self, x: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        Calculate a set of primitive internal coordinates (PIC) that are
        'close to' another set. This means that the restriction on dihedral
        angles being in the range (-π, π] is relaxed in favour of the smallest
        ∆q possible (where q is a value of a primitive coordinate).
        """
        assert len(self) == len(other) and isinstance(other, np.ndarray)

        q = self._calc_q(x)
        self._calc_B(x)

        for i, primitive in enumerate(self):
            if isinstance(primitive, DihedralAngle):

                dq = q[i] - other[i]

                if np.abs(dq) > np.pi:  # Ensure |dq| < π
                    q[i] -= np.sign(dq) * 2 * np.pi

        return q

    def __eq__(self, other: Any):
        """Comparison of two PIC sets"""

        is_equal = (
            isinstance(other, PIC)
            and len(other) == len(self)
            and all(p0 == p1 for p0, p1 in zip(self, other))
        )

        return is_equal

    def _calc_q(self, x: np.ndarray) -> np.ndarray:
        """Calculate the value of the internals"""

        if len(self) == 0:
            self._populate_all(x)

        return np.array([q(x) for q in self])

    @abstractmethod
    def _populate_all(self, x: np.ndarray) -> None:
        """Populate primitives from an array of cartesian coordinates"""

    def _calc_B(self, x: np.ndarray) -> None:
        """Calculate the Wilson B matrix"""

        if len(self) == 0:
            raise ValueError(
                "Cannot calculate the Wilson B matrix, no "
                "primitive internal coordinates"
            )

        cart_coords = x.reshape((-1, 3))

        n_atoms, _ = cart_coords.shape
        B = np.zeros(shape=(len(self), 3 * n_atoms))

        for i, primitive in enumerate(self):
            for j in range(n_atoms):

                B[i, 3 * j + 0] = primitive.derivative(
                    j, CartesianComponent.x, x=cart_coords
                )
                B[i, 3 * j + 1] = primitive.derivative(
                    j, CartesianComponent.y, x=cart_coords
                )
                B[i, 3 * j + 2] = primitive.derivative(
                    j, CartesianComponent.z, x=cart_coords
                )

        self._B = B
        return None

    @staticmethod
    def _are_all_primitive_coordinates(args: tuple) -> bool:
        return all(isinstance(arg, Primitive) for arg in args)

    @property
    def n_constrained(self) -> int:
        """Number of constrained primitive internal coordinates"""
        return sum(p.is_constrained for p in self)


class _FunctionOfDistances(PIC):
    @property
    @abstractmethod
    def _primitive_type(self) -> Type["_DistanceFunction"]:
        """Type of primitive coordinate defining f(r_ij)"""

    def _populate_all(self, x: np.ndarray):

        n_atoms = len(x.flatten()) // 3

        # Add all the unique inverse distances (i < j)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.append(self._primitive_type(i, j))

        return None


class InverseDistances(_FunctionOfDistances):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return InverseDistance


class Distances(_FunctionOfDistances):
    """r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return Distance
