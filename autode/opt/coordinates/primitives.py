import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Primitive(ABC):
    """Primitive internal coordinate"""

    @abstractmethod
    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates'
                 ) -> float:
        """Return the value of this PIC given a set of cartesian coordinates"""

    @abstractmethod
    def derivative(self,
                   i:         int,
                   component: str,
                   x:         'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        r"""
        Calculate the derivative with respect to a cartesian coordinate

        .. math::

            \frac{dq}
                  {d\boldsymbol{X}_{i, k}} {\Bigg\rvert}_{X=X0}

        where :math:`q` is the primitive coordinate and :math:`\boldsymbol{X}`
        are the cartesian coordinates.

        -----------------------------------------------------------------------
        Arguments:
            i: Cartesian index to take the derivative with respect to;
                0-N for N atoms

            component: Cartesian component (x, y, z) to take the derivative
                       with respect to. {'x', 'y', 'z'}

            x: Cartesian coordinates

        Returns:
            (float): Derivative
        """

    @staticmethod
    def _cart_component_to_idx(component: str) -> int:
        """Convert the name of a Cartesian component (or direction) to an
        index used for array indexing"""

        if component == 'x':
            return 0

        if component == 'y':
            return 1

        if component == 'z':
            return 2

        else:
            raise ValueError(f'Cannot convert component {component} to an '
                             f'index. Must be one of x, y, z')

    @abstractmethod
    def __eq__(self, other):
        """Comparison of two primitive coordinates"""


class _DistanceFunction(Primitive, ABC):
    """Function of a distance between two atoms"""

    def __init__(self,
                 idx_i: int,
                 idx_j: int):
        """
        Function of a distance between a pair of atoms

        .. math::

            q = f(|\boldsymbol{X}_i - \boldsymbol{X}_j|)

        for a set of cartesian coordinates :math:`\boldsymbol{X}`.
        -----------------------------------------------------------------------
        Arguments:
            idx_i: Atom index

            idx_j: Atom index
        """

        self.idx_i = int(idx_i)
        self.idx_j = int(idx_j)

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""

        return (isinstance(other, self.__class__)
                and other._ordered_idxs == self._ordered_idxs)

    @property
    def _ordered_idxs(self) -> Tuple[int, int]:
        """Indexes ordered by their value"""
        i, j = self.idx_i, self.idx_j

        return (i, j) if i < j else (j, i)


class InverseDistance(_DistanceFunction):
    """
    Inverse distance between to atoms:

    .. math::

        q = \frac{1}
                {|\boldsymbol{X}_i - \boldsymbol{X}_j|}
    """

    def derivative(self,
                   i:         int,
                   component: str,
                   x:        'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        """
        Derivative with respect to Cartesian displacement

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """

        _x = x.reshape((-1, 3))
        k = self._cart_component_to_idx(component)

        if i != self.idx_i and i != self.idx_j:
            return 0                 # Atom does not form part of this distance

        elif i == self.idx_i:
            return - (_x[i, k] - _x[self.idx_j, k]) * self(x)**3

        else:  # i == self.idx_j:
            return (_x[self.idx_i, k] - _x[self.idx_j, k]) * self(x)**3

    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates'
                 ) -> float:
        """1 / |x_i - x_j| """
        _x = x.reshape((-1, 3))
        return 1.0 / np.linalg.norm(_x[self.idx_i] - _x[self.idx_j])


class Distance(_DistanceFunction):
    """
    Distance between to atoms:

    .. math::

        q = |\boldsymbol{X}_i - \boldsymbol{X}_j|
    """

    def derivative(self,
                   i:         int,
                   component: str,
                   x:        'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        """
        Derivative with respect to Cartesian displacement

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """
        _x = x.reshape((-1, 3))
        k = self._cart_component_to_idx(component)

        if i != self.idx_i and i != self.idx_j:
            return 0                 # Atom does not form part of this distance

        val = (_x[self.idx_i, k] - _x[self.idx_j, k]) / self(x)

        return val if i == self.idx_i else -val

    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates'
                 ) -> float:
        """|x_i - x_j|"""
        _x = x.reshape((-1, 3))
        return np.linalg.norm(_x[self.idx_i] - _x[self.idx_j])


class ConstrainedDistance(Distance):

    def __init__(self,
                 idx_i: int,
                 idx_j: int,
                 value: float):
        """

        Args:
            idx_i:
            idx_j:
            value:
        """
        super().__init__(idx_i=idx_i, idx_j=idx_j)

        self._r0 = value
