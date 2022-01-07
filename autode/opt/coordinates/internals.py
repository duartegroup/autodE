"""
Internal coordinates. Notation follows


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
"""
import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.primitives import InverseDistance


class InternalCoordinates(OptCoordinates):

    def __new__(cls, input_array) -> 'InternalCoordinates':
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array, units=None)

        arr._x = None                # Cartesian coordinates
        arr.primitive_type = None    # Type of PIC e.g. InverseDistances

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        self._x = getattr(obj, '_x', None)
        self.primitive_type = getattr(obj, 'primitive_type', None)
        return OptCoordinates.__array_finalize__(self, obj)


class PIC(list, ABC):
    """Primitive internal coordinates"""

    def __init__(self,
                 x: 'autode.opt.CartesianCoordinates'):
        """
        Constructor for a set of primitives

        -----------------------------------------------------------------------
        Arguments:
            x: Cartesian coordinates
        """
        super().__init__()

        self._populate(x)                       # lgtm [py/init-calls-subclass]
        self.B = self._calc_B(x)
        self.q = np.array([p(x) for p in self])

    def __eq__(self, other: Any):
        """Comparison of two PIC sets"""

        is_equal = (isinstance(other, PIC)
                    and len(other) == len(self)
                    and all(p0 == p1 for p0, p1 in zip(self, other)))

        return is_equal

    @abstractmethod
    def _populate(self, x) -> None:
        """Populate Primitive-s used in the construction of set"""

    def _calc_B(self, x) -> np.ndarray:
        """Calculate the Wilson B matrix"""

        _x = x.reshape((-1, 3))

        n_atoms, _ = _x.shape
        B = np.zeros(shape=(len(self), 3 * n_atoms))

        for i, primitive in enumerate(self):
            for j in range(n_atoms):

                B[i, 3 * j + 0] = primitive.derivative(j, 'x', x=_x)
                B[i, 3 * j + 1] = primitive.derivative(j, 'y', x=_x)
                B[i, 3 * j + 2] = primitive.derivative(j, 'z', x=_x)

        return B


class InverseDistances(PIC):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    def _populate(self, x):

        n_atoms = len(x.flatten()) // 3

        # Add all the unique inverse distances (i < j)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.append(InverseDistance(i, j))

        return None
