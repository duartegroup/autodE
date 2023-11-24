# mypy: disable-error-code="has-type"
"""
Delocalised internal coordinate implementation from:
1. https://aip.scitation.org/doi/pdf/10.1063/1.478397
and references cited therein. Also used is
2. https://aip.scitation.org/doi/pdf/10.1063/1.1515483

The notation follows the paper and is briefly
summarised below:

| x : Cartesian coordinates
| B : Wilson B matrix
| G : 'Spectroscopic G matrix'
| q : Redundant internal coordinates
| s : Non-redundant internal coordinates
| U : Transformation matrix q -> s
"""
import numpy as np
from time import time
from typing import Optional, List, TYPE_CHECKING

from autode.geom import proj
from autode.log import logger
from autode.opt.coordinates.internals import (
    PIC,
    PrimitiveInverseDistances,
    InternalCoordinates,
)
from autode.exceptions import CoordinateTransformFailed

if TYPE_CHECKING:
    from autode.opt.coordinates import CartesianCoordinates, OptCoordinates
    from autode.values import Gradient
    from autode.hessians import Hessian


_max_back_transform_iterations = 20


class DIC(InternalCoordinates):  # lgtm [py/missing-equals]
    """Delocalised internal coordinates (DIC)"""

    def __repr__(self):
        return f"DIC(n={len(self)})"

    @staticmethod
    def _calc_U(primitives: PIC, x: "CartesianCoordinates") -> np.ndarray:
        r"""
        Transform matrix containing the non-redundant eigenvectors of the G
        matrix.

        .. math::

            G (U R) = (U R) \begin{pmatrix}
            \Lambda & 0 \\
            0 & 0
            \end{pmatrix}

        where

        .. math::

            G = B B^{T}


        -----------------------------------------------------------------------
        Arguments:
            primitives (autode.opt.internals.PIC):

        Returns:
            (np.ndarray): U
        """

        lambd, u = np.linalg.eigh(primitives.G)

        # Form a transform matrix from the primitive internals by removing the
        # redundant subspace comprised of small eigenvalues. This forms a set
        # of 3N - 6 non-redundant internals for a system of N atoms
        idxs = np.where(np.abs(lambd) > 1e-10)[0]

        if len(idxs) < x.expected_number_of_dof:
            raise RuntimeError(
                "Failed to create a complete set of delocalised internal "
                f"coordinates. {len(idxs)} < 3 N_atoms - 6. Likely due to "
                f"missing primitives"
            )

        logger.info(f"Removed {len(lambd) - len(idxs)} redundant vectors")
        return u[:, idxs]

    @classmethod
    def from_cartesian(
        cls,
        x: "CartesianCoordinates",
        primitives: Optional[PIC] = None,
    ) -> "DIC":
        """
        Convert cartesian coordinates to primitives then to delocalised
        internal coordinates (DICs), of which there should be 3N-6 for a
        polyatomic system with N atoms

        -----------------------------------------------------------------------
        Arguments:
            x: Cartesian coordinates

            primitives: Primitive internal coordinates. If undefined then use
                        all pairwise inverse distances

        Returns:
            (autode.opt.coordinates.DIC): Delocalised internal coordinates
        """
        logger.info("Converting cartesian coordinates to DIC")
        start_time = time()

        if primitives is None:
            logger.info("Building DICs from all inverse distances")
            primitives = PrimitiveInverseDistances.from_cartesian(x)

        q = primitives(x)
        U = cls._calc_U(primitives, x)
        U = _symmetry_inequivalent_u(U, q)

        dic = cls(input_array=np.matmul(U.T, q))

        dic.U = U  # Transform matrix primitives -> non-redundant

        dic.B = np.matmul(U.T, primitives.B)
        dic.B_T_inv = np.linalg.pinv(dic.B)
        dic._x = x.copy()
        dic.primitives = primitives

        dic.e = x.e  # Energy
        dic.update_g_from_cart_g(x.g)  # Gradient
        dic.update_h_from_cart_h(x.h)  # and Hessian

        logger.info(f"Transformed in      ...{time() - start_time:.4f} s")
        return dic

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient

        -----------------------------------------------------------------------
        Arguments:
            arr: Cartesian gradient array
        """
        if arr is None:
            self._x.g, self.g = None, None

        else:
            self._x.g = arr.flatten()
            self.g = np.matmul(self.B_T_inv.T, self._x.g)

        return None

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the DIC Hessian matrix from a Cartesian one

        -----------------------------------------------------------------------
        Arguments:
            arr: Cartesian Hessian matrix
        """
        if arr is None:
            self._x.h, self.h = None, None

        else:
            self._x.h = arr

            # NOTE: This is not the full transformation as noted in
            # 10.1063/1.471864 only an approximate Hessian is required(?)
            self.h = np.linalg.multi_dot((self.B_T_inv.T, arr, self.B_T_inv))

        return None

    def to(self, value: str) -> "OptCoordinates":
        """
        Convert these DICs to another type of coordinate

        -----------------------------------------------------------------------
        Arguments:
            value (str): e.g. "Cartesian"

        Returns:
            (autode.opt.coordinates.OptCoordinates): Coordinates
        """

        if value.lower() in ("x", "cart", "cartesian"):
            return self._x

        raise ValueError(f"Unknown conversion to {value}")

    def iadd(self, value: np.ndarray) -> "OptCoordinates":
        """
        Set some new internal coordinates and update the Cartesian coordinates

        .. math::

            x^(k+1) = x(k) + ({B^T})^{-1}(k)[s_{new} - s(k)]

        for an iteration k.

        ----------------------------------------------------------------------
        Keyword Arguments:

            value: Difference between the current and new DICs. Must be
                   the same shape as self into self.shape.
        Raises:
            (RuntimeError): If the transformation diverges
        """
        start_time = time()
        s_new = np.array(self, copy=True) + value

        # Initialise
        s_k, x_k = np.array(self, copy=True), self.to("cartesian").copy()
        q_init = self.primitives(x_k)
        x_1 = self.to("cartesian") + np.matmul(self.B_T_inv, value)

        success = False
        for i in range(1, _max_back_transform_iterations + 1):
            try:
                x_k = x_k + np.matmul(self.B_T_inv, (s_new - s_k))

                # Rebuild the primitives & DIC from the back-transformed Cartesians
                q_k = self.primitives.close_to(x_k, q_init)
                s_k = np.matmul(self.U.T, q_k)
                self.B = np.matmul(self.U.T, self.primitives.B)
                self.B_T_inv = np.linalg.pinv(self.B)

                rms_s = np.sqrt(np.mean(np.square(s_k - s_new)))

            # for ill-conditioned primitives, there might be math error
            except ArithmeticError:
                break

            if rms_s < 1e-10:
                success = True
                break

        if success:
            logger.info(
                f"DIC transformation converged in {i} cycle(s) "
                f"in {time() - start_time:.4f} s"
            )
        else:
            logger.warning(
                f"Failed to transform in {i} cycles. "
                f"Final RMS(s) = {rms_s:.8f}"
            )
            x_k = x_1
            if not self.allow_unconverged_back_transform:
                raise CoordinateTransformFailed(
                    "DIC->Cart iterative back-transform did not converge"
                )

        s_k = np.matmul(self.U.T, self.primitives(x_k))
        self.B = np.matmul(self.U.T, self.primitives.B)
        self.B_T_inv = np.linalg.pinv(self.B)

        self[:] = s_k
        self._x = x_k

        return self

    @property
    def active_indexes(self) -> List[int]:
        """A list of indexes for the active modes in this coordinate set"""
        return list(range(len(self)))

    @property
    def inactive_indexes(self) -> List[int]:
        """A list of indexes for the non-active modes in this coordinate set"""
        return []


class DICWithConstraints(DIC):
    r"""
    Delocalised internal coordinates (DIC) with constraints. Uses Lagrangian
    multipliers to enforce the constraints with:

    ..math::

        L(X, λ) = E(s) + \sum_{i=1}^m \lambda_i C_i(X)

    where s are internal coordinates, and C the constraint functions. The
    optimisation space then is the n non-constrained internal coordinates and
    the m Lagrangian multipliers (\lambda_i).
    """

    def __new__(cls, input_array) -> "InternalCoordinates":
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array)

        arr._lambda = None  # Additional lagrangian multipliers
        return arr

    def __array_finalize__(self, obj: "OptCoordinates") -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        super().__array_finalize__(obj)
        self._lambda = getattr(obj, "_lambda", None)
        return

    def zero_lagrangian_multipliers(self) -> None:
        r"""Zero all \lambda_i"""

        self._lambda = np.zeros(shape=(self.n_constraints,))
        return None

    @property
    def raw(self) -> np.ndarray:
        """Raw numpy array of these coordinates including the multipliers"""
        assert self._lambda is not None, "Must have λ defined"
        return np.array(self.tolist() + self._lambda.tolist(), copy=True)

    @staticmethod
    def _calc_U(primitives: PIC, x: "CartesianCoordinates") -> np.ndarray:
        """Eigenvectors of the G matrix"""

        u = DIC._calc_U(primitives, x)
        const_prim_idxs = [
            i
            for i, primitive in enumerate(primitives)
            if primitive.is_constrained
        ]

        logger.info(
            f"Projecting {len(const_prim_idxs)} constrained primitives"
        )
        return _schmidt_orthogonalise(u, *const_prim_idxs)

    @property
    def inactive_indexes(self) -> List[int]:
        """
        Generate a list of mode indexes that are inactive in the optimisation
        space. This *requires* the m constrained modes being at the end of the
        coordinate set. It also includes the lagrange multipliers
        """

        n, m = len(self), self.n_constraints
        x = self.to("cartesian")
        idxs = [
            i
            for i, p in enumerate(self.constrained_primitives)
            if p.is_satisfied(x)
        ]

        return [n - m + i for i in idxs] + [n + i for i in idxs]

    @property
    def active_indexes(self) -> List[int]:
        """Generate a list of indexes for the active modes in this coordinate
        set"""
        n, m = len(self), self.n_constraints  # n dic + m lagrange multipliers

        return [i for i in range(n + m) if i not in self.inactive_indexes]

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        r"""
        Updates the gradient from a calculated Cartesian gradient, where
        the gradient is that of the Lagrangian. Includes dL/d_λi terms where
        λi is the i-th lagrangian multiplier.

        -----------------------------------------------------------------------
        Arguments:
            arr: Cartesian gradient array
        """

        if arr is None:
            self._x.g, self.g = None, None
            return

        assert self._lambda is not None, "Must have λ defined"

        self._x.g = arr.flatten()
        n = len(self)
        m = self.n_constraints

        self.g = np.zeros(shape=(n + m,))

        # Set the first part dL/ds_i
        self.g[:n] = np.matmul(self.B_T_inv.T, self._x.g)

        for i in range(m):
            self.g[n - m + i] -= self._lambda[i] * 1  # λ dC_i/ds_i

        # and the final dL/dλ_i
        c = self.constrained_primitives
        for i in range(m):
            self.g[n + i] = -c[i].delta(self._x)  # C_i(x) = Z - Z_ideal

        return None

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the DIC Hessian matrix from a Cartesian one

        -----------------------------------------------------------------------
        Arguments:
            arr: Cartesian Hessian matrix
        """

        if arr is None:
            self._x.h, self.h = None, None

        else:
            self._x.h = arr
            n = len(self)
            m = self.n_constraints

            self.h = np.zeros(shape=(n + m, n + m))

            # Fill in the upper left corner with d^2L/ds_i ds_j
            # where the second derivative of the constraint is zero
            self.h[:n, :n] = np.linalg.multi_dot(
                (self.B_T_inv.T, arr, self.B_T_inv)
            )

            # and the d^2L/ds_i dλ_i = -dC_i/ds_i = -1
            #         d^2L/dλ_i dλ_j = 0

            for i in range(m):
                self.h[n + i, :] = self.h[:, n + i] = 0.0

            for i in range(m):
                self.h[n - m + i, n + i] = self.h[n + i, n - m + i] = -1.0

        return None

    def h_or_h_inv_has_correct_shape(self, arr: Optional[np.ndarray]):
        """Does a Hessian or its inverse have the correct shape?"""
        if arr is None:
            return True  # None is always valid

        n_rows, n_cols = arr.shape
        return (
            arr.ndim == 2
            and n_rows == n_cols == len(self) + self.n_constraints
        )

    def update_lagrange_multipliers(self, arr: np.ndarray) -> None:
        """Update the lagrange multipliers by adding a set of values"""
        assert self._lambda is not None, "Must have λ defined"

        if arr.shape != self._lambda.shape:
            raise ValueError(
                "Cannot set lagrange multipliers. Incorrect shape"
            )

        self._lambda[:] = np.asarray(self._lambda) + np.asarray(arr)
        return None


def _schmidt_orthogonalise(arr: np.ndarray, *indexes: int) -> np.ndarray:
    """
    Perform Schmidt orthogonalization to generate orthogonal vectors
    that include a number of unit vectors, the non-zero components of which
    are defined by indexes. This generates a transform matrix U which will
    provide pure primitive coordinates, which can then be constrained simply
    """
    logger.info(
        f"Schmidt-orthogonalizing. Using {indexes} as orthonormal vectors"
    )

    u = np.zeros_like(arr)
    _, n = arr.shape
    m = len(indexes)

    # Set the unit vectors as the first m columns
    for i, index in enumerate(indexes):
        u[index, i] = 1.0

    # and the remaining n-m columns as the orthogonalised values
    for i in range(m, n):
        u_i = arr[:, i]
        for j in range(0, i):
            u_i -= proj(u[:, j], arr[:, i])

        u_i /= np.linalg.norm(u_i)

        u[:, i] = u_i.copy()

    # Arbitrarily place the defined unit vectors at the end
    permutation = list(range(m, n)) + list(range(m))
    return u[:, permutation]


def _symmetry_inequivalent_u(u, q) -> np.ndarray:
    """Remove symmetry equivalent vectors from the U matrix"""

    # The non-redundant space can be further pruned by considering symmetry
    idxs: List[int] = []
    s = np.matmul(u.T, q)

    for i, s_i in enumerate(s):
        is_unique = all(not np.isclose(s_i, s[j], atol=1e-20) for j in idxs)

        if is_unique or _is_pure_primitive(u[:, i]):
            idxs.append(i)

    logger.info(f"Removing {len(s) - len(idxs)} symmetry equiv. DICs")
    return u[:, idxs]


def _is_pure_primitive(v: np.ndarray) -> bool:
    """
    Is this vector a pure primitive? Defined by all but one of the coefficients
    being zero
    """

    def n_values_close_to(value):
        return sum(np.isclose(v_i, value, atol=1e-10) for v_i in v)

    return n_values_close_to(0.0) == len(v) - 1 and n_values_close_to(1.0) == 1
