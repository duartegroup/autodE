import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, TYPE_CHECKING, List, Optional
from autode.opt.coordinates._autodiff import (
    get_differentiable_vars,
    DifferentiableMath,
    DifferentiableVector3D,
    DerivativeOrder,
    VectorHyperDual,
)

if TYPE_CHECKING:
    from autode.opt.coordinates import CartesianCoordinates


def _get_3d_vecs_from_atom_idxs(
    *args: int,
    x: "CartesianCoordinates",
    deriv_order: DerivativeOrder,
) -> List[DifferentiableVector3D]:
    """
    Obtain differentiable 3D vectors from the Cartesian components
    of each atom, given by atomic indices in order. The symbols are
    strings denoting their position in the flat Cartesian coordinate.

    Args:
        *args: Integers denoting the atom positions
        x: Cartesian coordinate array
        deriv_order: Order of derivatives for initialising variables

    Returns:
        (list[VectorHyperDual]): A list of differentiable variables
    """
    assert all(isinstance(idx, int) and idx >= 0 for idx in args)
    # get positions in the flat Cartesian array
    _x = x.ravel()
    cart_idxs = []
    for atom_idx in args:
        for k in range(3):
            cart_idxs.append(3 * atom_idx + k)
    variables = get_differentiable_vars(
        values=[_x[idx] for idx in cart_idxs],
        symbols=[str(idx) for idx in cart_idxs],
        deriv_order=deriv_order,
    )
    atom_vecs = []
    for pos_idx in range(len(args)):
        atom_vecs.append(
            DifferentiableVector3D(variables[pos_idx * 3 : pos_idx * 3 + 3])
        )
    return atom_vecs


class Primitive(ABC):
    """Primitive internal coordinate"""

    is_constrained = False

    def __init__(self, *atom_indexes: int):
        """A primitive internal coordinate that involves a number of atoms"""
        self._atom_indexes = atom_indexes

    @abstractmethod
    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ):
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
        res = self._evaluate(_x, deriv_order=DerivativeOrder.zeroth)
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
        res = self._evaluate(_x, deriv_order=DerivativeOrder.first)
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
        res = self._evaluate(_x, deriv_order=DerivativeOrder.second)
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
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ) -> "VectorHyperDual":
        """1 / |x_i - x_j|"""
        vec_i, vec_j = _get_3d_vecs_from_atom_idxs(
            self.i, self.j, x=x, deriv_order=deriv_order
        )
        return 1.0 / (vec_i - vec_j).norm()  # type: ignore

    def __repr__(self):
        return f"InverseDistance({self.i}-{self.j})"


class PrimitiveDistance(_DistanceFunction):
    r"""
    Distance between two atoms:

    .. math::

        q = |\boldsymbol{X}_i - \boldsymbol{X}_j|
    """

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ) -> "VectorHyperDual":
        """|x_i - x_j|"""
        vec_i, vec_j = _get_3d_vecs_from_atom_idxs(
            self.i, self.j, x=x, deriv_order=deriv_order
        )
        return (vec_i - vec_j).norm()  # type: ignore

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

    def __init__(self, m: int, o: int, n: int):
        """Bond angle m-o-n"""
        super().__init__(m, o, n)

        self.m = int(m)
        self.o = int(o)
        self.n = int(n)

    def __eq__(self, other) -> bool:
        """Equality of two distance functions"""

        return (
            isinstance(other, self.__class__)
            and self.o == other.o
            and other._ordered_idxs == self._ordered_idxs
        )

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ):
        """m - o - n angle"""
        vec_m, vec_o, vec_n = _get_3d_vecs_from_atom_idxs(
            self.m, self.o, self.n, x=x, deriv_order=deriv_order
        )
        u = vec_m - vec_o
        v = vec_n - vec_o
        return DifferentiableMath.acos(u.dot(v) / (u.norm() * v.norm()))

    def __repr__(self):
        return f"Angle({self.m}-{self.o}-{self.n})"


class ConstrainedPrimitiveBondAngle(ConstrainedPrimitive, PrimitiveBondAngle):
    def __init__(self, m: int, o: int, n: int, value: float):
        """
        Angle (m-o-n) constrained to a value (in radians)

        -----------------------------------------------------------------------
        Arguments:

            m: Atom index

            o: Atom index

            n: Atom index

            value: Required value of the constrained angle
        """
        super().__init__(m=m, o=o, n=n)

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
        """Equality of two dihedral angles"""
        return isinstance(other, self.__class__) and (
            self._atom_indexes == other._atom_indexes
            or self._atom_indexes == tuple(reversed(other._atom_indexes))
        )

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ) -> "VectorHyperDual":
        """Dihedral m-o-p-n"""
        # https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        _x = x.ravel()
        vec_m, vec_o, vec_p, vec_n = _get_3d_vecs_from_atom_idxs(
            self.m, self.o, self.p, self.n, x=_x, deriv_order=deriv_order
        )
        u_1 = vec_o - vec_m
        u_2 = vec_p - vec_o
        u_3 = vec_n - vec_p

        norm_u2 = u_2.norm()
        v1 = u_2.cross(u_3)
        v2 = u_1.cross(u_2)
        v3 = u_1 * norm_u2
        dihedral = DifferentiableMath.atan2(v3.dot(v1), v2.dot(v1))
        assert isinstance(dihedral, VectorHyperDual)
        return dihedral

    def __repr__(self):
        return f"Dihedral({self.m}-{self.o}-{self.p}-{self.n})"


class LinearBendType(Enum):
    """For linear angles, there are two orthogonal directions"""

    BEND = 0
    COMPLEMENT = 1


class LinearAngleBase(Primitive, ABC):
    def __init__(self, m: int, o: int, n: int, r: int, axis: LinearBendType):
        """Linear Bend: m-o-n"""
        super().__init__(m, o, n, r)
        self.m = int(m)
        self.o = int(o)
        self.n = int(n)
        self.r = int(r)

        assert isinstance(axis, LinearBendType)
        self.axis = axis

    def __eq__(self, other):
        """Equality of two linear bend angles"""
        return isinstance(other, self.__class__) and (
            self._atom_indexes == other._atom_indexes
            and self.axis == other.axis
        )

    def _calc_linear_bend(
        self,
        m_vec: DifferentiableVector3D,
        o_vec: DifferentiableVector3D,
        n_vec: DifferentiableVector3D,
        r_vec: DifferentiableVector3D,
    ):
        """
        Evaluate the linear bend from the vector positions of the
        atoms involved in the angle m, o, n, and the reference
        atom (or dummy atom) r

        Args:
            m_vec:
            o_vec:
            n_vec:
            r_vec:

        Returns:
            (VectorHyperDual): The value, optionally containing derivatives
        """
        # As defined in J. Comput. Chem., 20(10), 1999, 1067
        o_m = m_vec - o_vec
        o_n = n_vec - o_vec
        o_r = r_vec - o_vec

        # eq.(44) p 1073
        u = o_m.cross(o_r)
        u = u / u.norm()

        # eq. (46) and (47) p 1074
        if self.axis == LinearBendType.BEND:
            return u.dot(o_n) / o_n.norm()
        elif self.axis == LinearBendType.COMPLEMENT:
            return u.dot(o_n.cross(o_m)) / (o_n.norm() * o_m.norm())
        else:
            raise ValueError("Unknown axis for linear bend")


class PrimitiveLinearAngle(LinearAngleBase):
    """Linear Angle w.r.t. a reference atom"""

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ):
        """Linear Bend angle m-o-n against reference atom r"""

        _x = x.ravel()
        vec_m, vec_o, vec_n, vec_r = _get_3d_vecs_from_atom_idxs(
            self.m, self.o, self.n, self.r, x=_x, deriv_order=deriv_order
        )

        return self._calc_linear_bend(vec_m, vec_o, vec_n, vec_r)

    def __repr__(self):
        axis_str = "B" if self.axis == LinearBendType.BEND else "C"
        return f"LinearBend{axis_str}({self.m}-{self.o}-{self.n}, {self.r})"


class PrimitiveDummyLinearAngle(LinearAngleBase):
    """Linear bend with a dummy atom"""

    def __init__(self, m: int, o: int, n: int, axis: LinearBendType):
        super().__init__(m, o, n, -1, axis)

        self._vec_r: Optional[DifferentiableVector3D] = None

    def _get_dummy_atom(
        self, x: "CartesianCoordinates"
    ) -> DifferentiableVector3D:
        """Create the dummy atom r"""
        _x = x.reshape(-1, 3)
        cart_axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        # choose cartesian axis with the lowest overlap with m-o vector
        w = _x[self.m] - _x[self.o]
        w /= np.linalg.norm(w)
        overlaps = []
        for axis in cart_axes:
            overlaps.append(np.dot(w, axis))
        cart_ax = cart_axes[np.argmin(np.abs(overlaps))]

        # place dummy atom perpendicular to m-o bond 1 A away
        perp_axis = np.cross(cart_ax, w)
        _o = _x[self.o]
        dummy_point = _o + perp_axis / np.linalg.norm(perp_axis)
        return DifferentiableVector3D(list(dummy_point))

    def _evaluate(
        self, x: "CartesianCoordinates", deriv_order: DerivativeOrder
    ):
        if self._vec_r is None:
            self._vec_r = self._get_dummy_atom(x)

        _x = x.ravel()
        vec_m, vec_o, vec_n = _get_3d_vecs_from_atom_idxs(
            self.m, self.o, self.n, x=_x, deriv_order=deriv_order
        )

        return self._calc_linear_bend(vec_m, vec_o, vec_n, self._vec_r)

    def __repr__(self):
        axis_str = "B" if self.axis == LinearBendType.BEND else "C"
        return f"LinearBend{axis_str}({self.m}-{self.o}-{self.n}, D)"
