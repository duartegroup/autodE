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


def _norm_vec3(vec: List["VectorHyperDual"]) -> "VectorHyperDual":
    """Norm of a 3D vector"""
    assert len(vec) == 3
    return DifferentiableMath.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def _sub_vec3(
    vec1: List["VectorHyperDual"], vec2: List["VectorHyperDual"]
) -> List["VectorHyperDual"]:
    """Evaluate vec1 - vec2 for 3D vectors"""
    assert len(vec1) == len(vec2) == 3
    return [vec1[i] - vec2[i] for i in range(3)]


def _dot_vec3(
    vec1: List["VectorHyperDual"], vec2: List["VectorHyperDual"]
) -> "VectorHyperDual":
    """Evaluate vec1.vec2 for 3D vectors"""
    assert len(vec1) == len(vec2) == 3
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


def _cross_vec3(
    vec1: List["VectorHyperDual"], vec2: List["VectorHyperDual"]
) -> List["VectorHyperDual"]:
    """Evaluate vec1 x vec2 for 3D vectors"""
    assert len(vec1) == len(vec2) == 3
    # a cross b = a1 b2 - a2 b1, a2 b0 - a0 b2, a0 b1 - a1 b0
    return [
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    ]


def _get_vars_from_atom_idxs(
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
    assert all(isinstance(atom, int) and atom >= 0 for atom in args)
    # get positions in the flat Cartesian array
    _x = x.ravel()
    cart_idxs = []
    for atom in args:
        for k in range(3):
            cart_idxs.append(3 * atom + k)
    return get_differentiable_vars(
        values=[_x[idx] for idx in cart_idxs],
        symbols=[str(idx) for idx in cart_idxs],
        deriv_order=deriv_order,
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
        Calculate the derivatives with respect to cartesian coordinates

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

    def second_derivative(
        self,
        x: "CartesianCoordinates",
    ) -> np.ndarray:
        r"""
        Calculate the second derivatives with respect to cartesian coordinates

        .. math::

            \frac{d^2 q}
                  {d\boldsymbol{X}_{i, k}^2} {\Bigg\rvert}_{X=X0}

        where :math:`q` is the primitive coordinate and :math:`\boldsymbol{X}`
        are the cartesian coordinates.

        -----------------------------------------------------------------------
        Arguments:

            x: Cartesian coordinate array of shape (N, )

        Returns:
            (np.ndarray): Second derivative matrix of shape (N, N)
        """
        _x = x.ravel()
        x_n = _x.shape[0]
        res = self._evaluate(_x, deriv_order=2)
        derivs = np.zeros(shape=(x_n, x_n), dtype=float)
        for i in range(x_n):
            for j in range(x_n):
                d2q_dx2_ij = res.differentiate_wrt(str(i), str(j))
                if d2q_dx2_ij is not None:
                    derivs[i, j] = d2q_dx2_ij

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
    r"""
    Inverse distance between two atoms:

    .. math::

        q = \frac{1}
                {|\boldsymbol{X}_i - \boldsymbol{X}_j|}
    """

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: int
    ) -> "VectorHyperDual":
        """1 / |x_i - x_j|"""
        i_0, i_1, i_2, j_0, j_1, j_2 = _get_vars_from_atom_idxs(
            self.i, self.j, x=x, deriv_order=deriv_order
        )
        return 1.0 / _norm_vec3(_sub_vec3([i_0, i_1, i_2], [j_0, j_1, j_2]))

    def __repr__(self):
        return f"InverseDistance({self.i}-{self.j})"


class PrimitiveDistance(_DistanceFunction):
    r"""
    Distance between two atoms:

    .. math::

        q = |\boldsymbol{X}_i - \boldsymbol{X}_j|
    """

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: int
    ) -> "VectorHyperDual":
        """|x_i - x_j|"""
        i_0, i_1, i_2, j_0, j_1, j_2 = _get_vars_from_atom_idxs(
            self.i, self.j, x=x, deriv_order=deriv_order
        )
        return _norm_vec3(_sub_vec3([i_0, i_1, i_2], [j_0, j_1, j_2]))

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
    """
    Bond angle between three atoms, calculated with the
    arccosine of the normalised dot product
    """

    def __init__(self, o: int, m: int, n: int):
        """Bond angle m-o-n"""
        super().__init__(o, m, n)

        self.o = int(o)
        self.m = int(m)
        self.n = int(n)

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""

        return (
            isinstance(other, self.__class__)
            and self.o == other.o
            and other._ordered_idxs == self._ordered_idxs
        )

    def _evaluate(self, x: "CartesianCoordinates", deriv_order: int):
        """m - o - n angle"""
        variables = _get_vars_from_atom_idxs(
            self.m, self.o, self.n, x=x, deriv_order=deriv_order
        )
        _m = variables[0:3]
        _o = variables[3:6]
        _n = variables[6:9]
        u = _sub_vec3(_m, _o)
        v = _sub_vec3(_n, _o)
        theta = DifferentiableMath.acos(
            _dot_vec3(u, v) / (_norm_vec3(u) * _norm_vec3(v))
        )
        return theta

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

        self.m = int(m)
        self.o = int(o)
        self.p = int(p)
        self.n = int(n)

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""
        return isinstance(other, self.__class__) and (
            self._atom_indexes == other._atom_indexes
            or self._atom_indexes == tuple(reversed(other._atom_indexes))
        )

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: int
    ) -> "VectorHyperDual":
        """Dihedral m-o-p-n"""
        # https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        _x = x.ravel()
        variables = _get_vars_from_atom_idxs(
            self.m, self.o, self.p, self.n, x=_x, deriv_order=deriv_order
        )
        _m = variables[:3]
        _o = variables[3:6]
        _p = variables[6:9]
        _n = variables[9:12]

        u_1 = _sub_vec3(_o, _m)
        u_2 = _sub_vec3(_p, _o)
        u_3 = _sub_vec3(_n, _p)

        norm_u2 = _norm_vec3(u_2)
        v1 = _cross_vec3(u_2, u_3)
        v2 = _cross_vec3(u_1, u_2)
        v3 = [k * norm_u2 for k in u_1]
        return DifferentiableMath.atan2(_dot_vec3(v3, v1), _dot_vec3(v2, v1))

    def __repr__(self):
        return f"Dihedral({self.m}-{self.o}-{self.p}-{self.n})"
