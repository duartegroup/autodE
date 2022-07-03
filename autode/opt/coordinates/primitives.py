import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List


class Primitive(ABC):
    """Primitive internal coordinate"""

    is_constrained = False

    @abstractmethod
    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates'
                 ) -> float:
        """Return the value of this PIC given a set of cartesian coordinates"""

    @abstractmethod
    def derivative(self,
                   i:         int,
                   component: 'autode.opt.coordinates.CartesianComponent',
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
            i: Cartesian index to take the derivative with respect to. [0-N),
               for N atoms

            component: Cartesian component (x, y, z) to take the derivative
                       with respect to

            x: Cartesian coordinates

        Returns:
            (float): Derivative
        """

    @abstractmethod
    def __eq__(self, other):
        """Comparison of two primitive coordinates"""


class ConstrainedPrimitive(Primitive, ABC):

    is_constrained = True

    @property
    @abstractmethod
    def _value(self) -> float:
        """Value of the constraint that must be satisfied e.g. r0"""

    def is_satisfied(self,
                     x:   'autode.opt.coordinates.CartesianCoordinates',
                     tol: float = 1E-4
                     ) -> bool:
        """Is this constraint satisfied to within an absolute tolerance"""
        return abs(self.delta(x)) < tol

    def delta(self,
              x: 'autode.opt.coordinates.CartesianCoordinates',
              ) -> float:
        """Difference between the observed and required value"""
        return self(x) - self._value


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
                   i:          int,
                   component: 'autode.opt.coordinates.CartesianComponent',
                   x:         'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        """
        Derivative with respect to Cartesian displacement

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """

        _x = x.reshape((-1, 3))
        k = int(component)

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
    Distance between two atoms:

    .. math::

        q = |\boldsymbol{X}_i - \boldsymbol{X}_j|
    """

    def derivative(self,
                   i:          int,
                   component: 'autode.opt.coordinates.CartesianComponent',
                   x:         'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        """
        Derivative with respect to Cartesian displacement

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """
        _x = x.reshape((-1, 3))
        k = int(component)

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


class ConstrainedDistance(ConstrainedPrimitive, Distance):

    def __init__(self,
                 idx_i: int,
                 idx_j: int,
                 value: float):
        """
        Distance constrained to a value

        -----------------------------------------------------------------------
        Arguments:

            idx_i: Atom index of the first atom

            idx_j: Atom index of the second atom

            value: Required value of the constrained distance
        """
        super().__init__(idx_i=idx_i, idx_j=idx_j)

        self._r0 = value

    @property
    def _value(self) -> float:
        return self._r0


class BondAngle(Primitive):

    def __init__(self, o: int, m: int, n: int):
        self.o = o
        self.m = m
        self.n = n

    @property
    def _ordered_idxs(self) -> List[int]:
        return list(sorted([self.o, self.m, self.n]))

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""

        return (isinstance(other, self.__class__)
                and other._ordered_idxs == self._ordered_idxs)

    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates') -> float:

        _x = x.reshape((-1, 3))
        u = _x[self.m, :] - _x[self.o, :]
        v = _x[self.n, :] - _x[self.o, :]

        theta = np.arccos(u.dot(v)
                          / (np.linalg.norm(u) * np.linalg.norm(v)))
        return theta

    def derivative(self,
                   i: int,
                   component: 'autode.opt.coordinates.CartesianComponent',
                   x: 'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:

        if i not in (self.o, self.m, self.n):
            return 0.

        k = int(component)

        _x = x.reshape((-1, 3))
        u = _x[self.m, :] - _x[self.o, :]
        lambda_u = np.linalg.norm(u)
        u /= lambda_u

        v = _x[self.n, :] - _x[self.o, :]
        lambda_v = np.linalg.norm(v)
        v /= lambda_v

        t0, t1 = np.array([1., -1., 1.]), np.array([-1., 1., 1.])

        if not np.isclose(np.arccos(u.dot(v)), 0.):
            w = np.cross(u, v)
        elif (not np.isclose(np.arccos(u.dot(t0)), 0.)
              and not np.isclose(np.arccos(v.dot(t0)), 0.)):
            w = np.cross(u, t0)
        else:
            w = np.cross(u, t1)

        w /= np.linalg.norm(w)

        dqdx = 0.

        if i in (self.m, self.o):
            sign = 1 if i == self.m else -1
            dqdx += sign * np.cross(u, w)[k] / lambda_u

        if i in (self.n, self.o):
            sign = 1 if i == self.n else -1
            dqdx += sign * np.cross(w, v)[k] / lambda_v

        return dqdx
