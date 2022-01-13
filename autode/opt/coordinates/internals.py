"""
Internal coordinates. Notation follows:


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
G : Spectroscopic G matrix
"""
import numpy as np
from typing import Any, Optional
from abc import ABC, abstractmethod
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.primitives import InverseDistance, Primitive


class InternalCoordinates(OptCoordinates, ABC):   # lgtm [py/missing-equals]

    def __new__(cls, input_array) -> 'InternalCoordinates':
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array, units=None)

        arr._x = None           # Cartesian coordinates
        arr.primitives = None   # PIC

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        self._x = getattr(obj, '_x', None)
        self.primitives = getattr(obj, 'primitives', None)
        return OptCoordinates.__array_finalize__(self, obj)


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
            raise ValueError('Cannot construct primitive internal coordinates '
                             f'from {args}. Must be primitive internals')

    @property
    def B(self) -> np.ndarray:
        """Wilson B matrix"""

        if self._B is None:
            raise AttributeError(f'{self} had no B matrix. Please calculate '
                                 f'the value of the primitives to determine B')

        return self._B

    @property
    def G(self) -> np.ndarray:
        """Spectroscopic G matrix as the symmetrised Wilson B matrix"""
        return np.dot(self.B, self.B.T)

    @classmethod
    def from_cartesian(cls,
                       x:          'autode.opt.cartesian.CartesianCoordinates',
                       ) -> 'PIC':
        """Construct a complete set of primitive internal coordinates from
        a set of Cartesian coordinates"""

        pic = cls()
        pic._populate_all(x=x)

        return pic

    def __eq__(self, other: Any):
        """Comparison of two PIC sets"""

        is_equal = (isinstance(other, PIC)
                    and len(other) == len(self)
                    and all(p0 == p1 for p0, p1 in zip(self, other)))

        return is_equal

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Populate Primitive-s used in the construction of set"""

        q = self._calc_q(x)
        self._calc_B(x)

        return q

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
            raise ValueError('Cannot calculate the Wilson B matrix, no '
                             'primitive internal coordinates')

        _x = x.reshape((-1, 3))

        n_atoms, _ = _x.shape
        B = np.zeros(shape=(len(self), 3 * n_atoms))

        for i, primitive in enumerate(self):
            for j in range(n_atoms):

                B[i, 3 * j + 0] = primitive.derivative(j, 'x', x=_x)
                B[i, 3 * j + 1] = primitive.derivative(j, 'y', x=_x)
                B[i, 3 * j + 2] = primitive.derivative(j, 'z', x=_x)

        self._B = B
        return None

    @staticmethod
    def _are_cartesian_coordinates(args: tuple) -> bool:
        """Are a tuple of arguments Cartesian coordinates?"""
        from autode.opt.coordinates.cartesian import CartesianCoordinates
        return len(args) == 1 and isinstance(args[0], CartesianCoordinates)

    @staticmethod
    def _are_all_primitive_coordinates(args: tuple) -> bool:
        return all(isinstance(arg, Primitive) for arg in args)


class InverseDistances(PIC):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    def _populate_all(self, x: np.ndarray):

        n_atoms = len(x.flatten()) // 3

        # Add all the unique inverse distances (i < j)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.append(InverseDistance(i, j))

        return None
