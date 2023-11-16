import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING, List, Optional
from autode.opt.coordinates.autodiff import (
    get_differentiable_vars,
    DifferentiableMath,
)

if TYPE_CHECKING:
    from autode.opt.coordinates import CartesianCoordinates, CartesianComponent
    from autode.opt.coordinates.autodiff import VectorHyperDual


def _norm_vec3(vec: List["VectorHyperDual"]):
    """Norm of a 3D vector"""
    assert len(vec) == 3
    return DifferentiableMath.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def _sub_vec3(vec1: List["VectorHyperDual"], vec2: List["VectorHyperDual"]):
    """Evaluate vec1 - vec2 for 3D vectors"""
    assert len(vec1) == len(vec2) == 3
    return [vec1[i] - vec2[i] for i in range(3)]


def _dot_vec3(vec1: List["VectorHyperDual"], vec2: List["VectorHyperDual"]):
    """Evaluate vec1.vec2 for 3D vectors"""
    assert len(vec1) == len(vec2) == 3
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def get_vars_from_atom_idxs(
    *args,
    x: "CartesianCoordinates",
    deriv_order: int,
) -> List["VectorHyperDual"]:
    """
    Obtain differentiable variables from the Cartesian components
    of each atom, given by atomic indices in order. The symbols are
    strings denoting their position in the flat Cartesian coordinate.

    Args:
        *args: Integers denoting the atom positions
        x: Cartesian coordinate array
        deriv_order: Order of derivatives for initialising variables

    Returns:
        (list[VectorHyperDual]): A list of differentiable variables
    """
    assert all(isinstance(atom, int) and atom > 0 for atom in args)
    # get positions in the flat Cartesian array
    _x = x.ravel()
    cart_idxs = []
    for atom in args:
        for k in range(3):
            cart_idxs.append(3 * atom + k)
    return get_differentiable_vars(
        values=[_x[idx] for idx in cart_idxs],
        deriv_order=deriv_order,
        symbols=[str(idx) for idx in cart_idxs],
    )


class Primitive(ABC):
    """Primitive internal coordinate"""

    is_constrained = False

    def __init__(self, *atom_indexes: int):
        """A primitive internal coordinate that involves a number of atoms"""
        self._atom_indexes = atom_indexes

    def _evaluate(self, x: "CartesianCoordinates", deriv_order: int):
        """
        The function that performs the main evaluation of the PIC,
        and optionally returns derivative or second derivatives.
        The returned hyper-dual must have the proper cartesian idxs
        set.

        Args:
            x: Cartesian coordinates
            deriv_order: The order of derivatives requested - 0, 1 or 2

        Returns:
            (VectorHyperDual): The result, optionally containing derivatives
        """

    @abstractmethod
    def __call__(self, x: "CartesianCoordinates") -> float:
        """Return the value of this PIC given a set of cartesian coordinates"""
        _x = x.ravel()
        res = self._evaluate(_x, deriv_order=0)
        return res.value

    def derivative(
        self,
        x: "CartesianCoordinates",
    ) -> np.ndarray:
        r"""
        Calculate the derivative with respect to a cartesian coordinate

        .. math::

            \frac{dq}
                  {d\boldsymbol{X}_{i, k}} {\Bigg\rvert}_{X=X0}

        where :math:`q` is the primitive coordinate and :math:`\boldsymbol{X}`
        are the cartesian coordinates.

        -----------------------------------------------------------------------
        Arguments:

            x: Cartesian coordinate array of shape (N, )

        Returns:
            (np.ndarray): Derivative array of shape (N, )
        """
        _x = x.ravel()
        res = self._evaluate(_x, deriv_order=1)
        derivs = np.zeros_like(_x, dtype=float)
        for i in range(_x.shape[0]):
            dqdx_i = res.differentiate_wrt(str(i))
            if dqdx_i is not None:
                derivs[i] = dqdx_i

        return derivs

    @abstractmethod
    def __eq__(self, other):
        """Comparison of two primitive coordinates"""

    @property
    def _ordered_idxs(self) -> Tuple[int, ...]:
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
        x: "CartesianCoordinates",
        tol: float = 1e-4,
    ) -> bool:
        """Is this constraint satisfied to within an absolute tolerance"""
        return abs(self.delta(x)) < tol

    def delta(
        self,
        x: "CartesianCoordinates",
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


class PrimitiveInverseDistance(_DistanceFunction):
    """
    Inverse distance between to atoms:

    .. math::

        q = \frac{1}
                {|\boldsymbol{X}_i - \boldsymbol{X}_j|}
    """

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: int
    ) -> "VectorHyperDual":
        """1 / |x_i - x_j|"""
        i_0, i_1, i_2, j_0, j_1, j_2 = get_vars_from_atom_idxs(
            self.i, self.j, x=x, deriv_order=deriv_order
        )
        return 1.0 / _norm_vec3(_sub_vec3([i_0, i_1, i_2], [j_0, j_1, j_2]))

    def legacy_derivative(
        self,
        i: int,
        component: "CartesianComponent",
        x: "CartesianCoordinates",
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

    def __call__(self, x: "CartesianCoordinates") -> float:
        """1 / |x_i - x_j|"""
        _x = x.reshape((-1, 3))
        return 1.0 / np.linalg.norm(_x[self.i] - _x[self.j])


class PrimitiveDistance(_DistanceFunction):
    """
    Distance between two atoms:

    .. math::

        q = |\boldsymbol{X}_i - \boldsymbol{X}_j|
    """

    def legacy_derivative(
        self,
        i: int,
        component: "CartesianComponent",
        x: "CartesianCoordinates",
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

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: int
    ) -> "VectorHyperDual":
        """|x_i - x_j|"""
        i_0, i_1, i_2, j_0, j_1, j_2 = get_vars_from_atom_idxs(
            self.i, self.j, x=x, deriv_order=deriv_order
        )
        return _norm_vec3(_sub_vec3([i_0, i_1, i_2], [j_0, j_1, j_2]))

    def __call__(self, x: "CartesianCoordinates") -> float:
        """|x_i - x_j|"""
        _x = x.reshape((-1, 3))
        return np.linalg.norm(_x[self.i] - _x[self.j])

    def __repr__(self):
        return f"Distance({self.i}-{self.j})"


class ConstrainedPrimitiveDistance(ConstrainedPrimitive, PrimitiveDistance):
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


class PrimitiveBondAngle(Primitive):
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

    def _evaluate(self, x: "CartesianCoordinates", deriv_order: int):
        """m - o - n angle"""
        m_0, m_1, m_2, o_0, o_1, o_2, n_0, n_1, n_2 = get_vars_from_atom_idxs(
            self.m, self.o, self.n, x=x, deriv_order=deriv_order
        )
        u = _sub_vec3([m_0, m_1, m_2], [o_0, o_1, o_2])
        v = _sub_vec3([n_0, n_1, n_2], [o_0, o_1, o_2])
        theta = DifferentiableMath.acos(
            _dot_vec3(u, v) / (_norm_vec3(u) * _norm_vec3(v))
        )
        return theta

    def __call__(self, x: "CartesianCoordinates") -> float:
        _x = x.reshape((-1, 3))
        u = _x[self.m, :] - _x[self.o, :]
        v = _x[self.n, :] - _x[self.o, :]

        theta = np.arccos(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return theta

    def legacy_derivative(
        self,
        i: int,
        component: "CartesianComponent",
        x: "CartesianCoordinates",
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


class ConstrainedPrimitiveBondAngle(ConstrainedPrimitive, PrimitiveBondAngle):
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

    def __eq__(self, other: object):
        return (
            super().__eq__(other)
            and isinstance(other, ConstrainedPrimitiveBondAngle)
            and np.isclose(self._theta0, other._theta0)
        )


class PrimitiveDihedralAngle(Primitive):
    def __init__(self, m: int, o: int, p: int, n: int):
        """Dihedral angle: m-o-p-n"""
        super().__init__(m, o, p, n)

        self.m = m
        self.o = o
        self.p = p
        self.n = n

    def __call__(self, x: "CartesianCoordinates") -> float:
        """Value of the dihedral"""
        return self._value(x, return_derivative=False)

    def legacy_derivative(
        self,
        i: int,
        component: "CartesianComponent",
        x: "CartesianCoordinates",
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
