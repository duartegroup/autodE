import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple


class Primitive(ABC):
    """Primitive internal coordinate"""

    is_constrained = False

    def __init__(self, *atom_indexes: int):
        """A primitive internal coordinate that involves a number of atoms"""
        self._atom_indexes = atom_indexes

    @abstractmethod
    def __call__(
        self, x: "autode.opt.coordinates.CartesianCoordinates"
    ) -> float:
        """Return the value of this PIC given a set of cartesian coordinates"""

    @abstractmethod
    def derivative(
        self,
        i: int,
        component: "autode.opt.coordinates.CartesianComponent",
        x: "autode.opt.coordinates.CartesianCoordinates",
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

    @property
    def _ordered_idxs(self) -> Tuple[int]:
        """Atom indexes ordered smallest to largest"""
        return tuple(sorted(self._atom_indexes))


class ConstrainedPrimitive(Primitive, ABC):
    """A primitive internal coordinate constrained to a value"""

    is_constrained = True

    @property
    @abstractmethod
    def _value(self) -> float:
        """Value of the constraint that must be satisfied e.g. r0"""

    def is_satisfied(
        self,
        x: "autode.opt.coordinates.CartesianCoordinates",
        tol: float = 1e-4,
    ) -> bool:
        """Is this constraint satisfied to within an absolute tolerance"""
        return abs(self.delta(x)) < tol

    def delta(
        self,
        x: "autode.opt.coordinates.CartesianCoordinates",
    ) -> float:
        """Difference between the observed and required value"""
        return self(x) - self._value


class _DistanceFunction(Primitive, ABC):
    """Function of a distance between two atoms"""

    def __init__(self, i: int, j: int):
        """
        Function of a distance between a pair of atoms

        .. math::

            q = f(|\boldsymbol{X}_i - \boldsymbol{X}_j|)

        for a set of cartesian coordinates :math:`\boldsymbol{X}`.
        -----------------------------------------------------------------------
        Arguments:
            i: Atom index

            j: Atom index
        """
        super().__init__(i, j)

        self.i = int(i)
        self.j = int(j)

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""

        return (
            isinstance(other, self.__class__)
            and other._ordered_idxs == self._ordered_idxs
        )


class InverseDistance(_DistanceFunction):
    """
    Inverse distance between to atoms:

    .. math::

        q = \frac{1}
                {|\boldsymbol{X}_i - \boldsymbol{X}_j|}
    """

    def derivative(
        self,
        i: int,
        component: "autode.opt.coordinates.CartesianComponent",
        x: "autode.opt.coordinates.CartesianCoordinates",
    ) -> float:
        """
        Derivative with respect to Cartesian displacement

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """

        _x = x.reshape((-1, 3))
        k = int(component)

        if i != self.i and i != self.j:
            return 0  # Atom does not form part of this distance

        elif i == self.i:
            return -(_x[i, k] - _x[self.j, k]) * self(x) ** 3

        else:  # i == self.idx_j:
            return (_x[self.i, k] - _x[self.j, k]) * self(x) ** 3

    def __call__(
        self, x: "autode.opt.coordinates.CartesianCoordinates"
    ) -> float:
        """1 / |x_i - x_j|"""
        _x = x.reshape((-1, 3))
        return 1.0 / np.linalg.norm(_x[self.i] - _x[self.j])


class Distance(_DistanceFunction):
    """
    Distance between two atoms:

    .. math::

        q = |\boldsymbol{X}_i - \boldsymbol{X}_j|
    """

    def derivative(
        self,
        i: int,
        component: "autode.opt.coordinates.CartesianComponent",
        x: "autode.opt.coordinates.CartesianCoordinates",
    ) -> float:
        """
        Derivative with respect to Cartesian displacement

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """
        _x = x.reshape((-1, 3))
        k = int(component)

        if i != self.i and i != self.j:
            return 0  # Atom does not form part of this distance

        val = (_x[self.i, k] - _x[self.j, k]) / self(x)

        return val if i == self.i else -val

    def __call__(
        self, x: "autode.opt.coordinates.CartesianCoordinates"
    ) -> float:
        """|x_i - x_j|"""
        _x = x.reshape((-1, 3))
        return np.linalg.norm(_x[self.i] - _x[self.j])

    def __repr__(self):
        return f"Distance({self.i}-{self.j})"


class ConstrainedDistance(ConstrainedPrimitive, Distance):
    def __init__(self, i: int, j: int, value: float):
        """
        Distance constrained to a value

        -----------------------------------------------------------------------
        Arguments:

            i: Atom index of the first atom

            j: Atom index of the second atom

            value: Required value of the constrained distance
        """
        super().__init__(i=i, j=j)

        self._r0 = value

    @property
    def _value(self) -> float:
        return self._r0

    def __repr__(self):
        return f"ConstrainedDistance({self.i}-{self.j})"


class BondAngle(Primitive):
    def __init__(self, o: int, m: int, n: int):
        """Bond angle m-o-n"""
        super().__init__(o, m, n)

        self.o = o
        self.m = m
        self.n = n

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""

        return (
            isinstance(other, self.__class__)
            and self.o == other.o
            and other._ordered_idxs == self._ordered_idxs
        )

    def __call__(
        self, x: "autode.opt.coordinates.CartesianCoordinates"
    ) -> float:

        _x = x.reshape((-1, 3))
        u = _x[self.m, :] - _x[self.o, :]
        v = _x[self.n, :] - _x[self.o, :]

        theta = np.arccos(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return theta

    def derivative(
        self,
        i: int,
        component: "autode.opt.coordinates.CartesianComponent",
        x: "autode.opt.coordinates.CartesianCoordinates",
    ) -> float:

        if i not in (self.o, self.m, self.n):
            return 0.0

        k = int(component)

        _x = x.reshape((-1, 3))
        u = _x[self.m, :] - _x[self.o, :]
        lambda_u = np.linalg.norm(u)
        u /= lambda_u

        v = _x[self.n, :] - _x[self.o, :]
        lambda_v = np.linalg.norm(v)
        v /= lambda_v

        t0, t1 = np.array([1.0, -1.0, 1.0]), np.array([-1.0, 1.0, 1.0])

        if not np.isclose(np.abs(np.arccos(u.dot(v))), 1.0):
            w = np.cross(u, v)
        elif not np.isclose(
            np.abs(np.arccos(u.dot(t0))), 1.0
        ) and not np.isclose(np.abs(np.arccos(v.dot(t0))), 1.0):
            w = np.cross(u, t0)
        else:
            w = np.cross(u, t1)

        w /= np.linalg.norm(w)

        dqdx = 0.0

        if i in (self.m, self.o):
            sign = 1 if i == self.m else -1
            dqdx += sign * np.cross(u, w)[k] / lambda_u

        if i in (self.n, self.o):
            sign = 1 if i == self.n else -1
            dqdx += sign * np.cross(w, v)[k] / lambda_v

        return dqdx

    def __repr__(self):
        return f"Angle({self.m}-{self.o}-{self.n})"


class ConstrainedBondAngle(ConstrainedPrimitive, BondAngle):
    def __init__(self, o: int, m: int, n: int, value: float):
        """
        Angle (m-o-n) constrained to a value (in radians)

        -----------------------------------------------------------------------
        Arguments:

            o: Atom index

            m: Atom index

            n: Atom index

            value: Required value of the constrained angle
        """
        super().__init__(o=o, m=m, n=n)

        self._theta0 = value

    @property
    def _value(self) -> float:
        return self._theta0

    def __repr__(self):
        return f"ConstrainedCAngle({self.m}-{self.o}-{self.n})"

    def __eq__(self, other):
        return super().__eq__(other) and np.isclose(
            self._theta0, other._theta0
        )


class DihedralAngle(Primitive):
    def __init__(self, m: int, o: int, p: int, n: int):
        """Dihedral angle: m-o-p-n"""
        super().__init__(m, o, p, n)

        self.m = m
        self.o = o
        self.p = p
        self.n = n

    def __call__(
        self, x: "autode.opt.coordinates.CartesianCoordinates"
    ) -> float:
        """Value of the dihedral"""
        return self._value(x, return_derivative=False)

    def derivative(
        self,
        i: int,
        component: "autode.opt.coordinates.CartesianComponent",
        x: "autode.opt.coordinates.CartesianCoordinates",
    ) -> float:
        return self._value(x, i=i, component=component, return_derivative=True)

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""
        return isinstance(other, self.__class__) and (
            self._atom_indexes == other._atom_indexes
            or self._atom_indexes == tuple(reversed(other._atom_indexes))
        )

    def _value(self, x, i=None, component=None, return_derivative=False):
        """Evaluate either the value or the derivative. Shared function
        to reuse local variables"""

        _x = x.reshape((-1, 3))
        u = _x[self.m, :] - _x[self.o, :]
        lambda_u = np.linalg.norm(u)
        u /= lambda_u

        v = _x[self.n, :] - _x[self.p, :]
        lambda_v = np.linalg.norm(v)
        v /= lambda_v

        w = _x[self.p, :] - _x[self.o, :]
        lambda_w = np.linalg.norm(w)
        w /= lambda_w

        phi_u = np.arccos(u.dot(w))
        phi_v = np.arccos(w.dot(v))

        if not return_derivative:
            v1 = np.cross(u, w)
            v2 = np.cross(-w, v)
            return -np.arctan2(np.cross(v1, w).dot(v2), (v1.dot(v2)))

        # are now computing the derivative..
        if i not in self._atom_indexes:
            return 0.0

        k = int(component)
        dqdx = 0.0

        if i in (self.m, self.o):
            sign = 1 if i == self.m else -1
            dqdx += sign * (
                np.cross(u, w)[k] / (lambda_u * np.sin(phi_u) ** 2)
            )

        if i in (self.p, self.n):
            sign = 1 if i == self.p else -1
            dqdx += sign * (
                np.cross(v, w)[k] / (lambda_v * np.sin(phi_v) ** 2)
            )

        if i in (self.o, self.p):
            sign = 1 if i == self.o else -1
            dqdx += sign * (
                (
                    (np.cross(u, w)[k] * np.cos(phi_u))
                    / (lambda_w * np.sin(phi_u) ** 2)
                )
                - (
                    (np.cross(v, w)[k] * np.cos(phi_v))
                    / (lambda_w * np.sin(phi_v) ** 2)
                )
            )

        return dqdx

    def __repr__(self):
        return f"Dihedral({self.m}-{self.o}-{self.p}-{self.n})"
