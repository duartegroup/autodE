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


MAX_BACK_TRANSFORM_ITERS = 50


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
        # calculate spectroscopic G matrix
        B = primitives.get_B(x)
        G = np.dot(B, B.T)
        lambd, u = np.linalg.eigh(G)

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

        dic.B = np.matmul(U.T, primitives.get_B(x))
        dic.B_T_inv = np.linalg.pinv(dic.B)
        dic._q = q.copy()
        dic._x = x.copy()
        dic.primitives = primitives

        dic.e = x.e  # Energy
        dic.update_g_from_cart_g(x.g)  # Gradient
        dic.update_h_from_cart_h(x.h)  # and Hessian

        logger.info(f"Transformed in      ...{time() - start_time:.4f} s")
        return dic

    @property
    def cart_proj_g(self) -> Optional[np.ndarray]:
        return self.to("cart").g

    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient

        -----------------------------------------------------------------------
        Arguments:
            arr: Cartesian gradient array
        """
        if arr is None:
            self._x.g, self._g = None, None

        else:
            self._x.g = arr.flatten()
            self._g = np.matmul(self.B_T_inv.T, self._x.g)

        return None

    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """
        Update the DIC Hessian matrix from a Cartesian one

        -----------------------------------------------------------------------
        Arguments:
            arr: Cartesian Hessian matrix
        """
        if arr is None:
            self._x.h, self._h = None, None

        else:
            self._x.h = arr

            # NOTE: This is not the full transformation as noted in
            # 10.1063/1.471864 only an approximate Hessian is required(?)
            hess = np.linalg.multi_dot((self.B_T_inv.T, arr, self.B_T_inv))
            assert self.h_or_h_inv_has_correct_shape(hess)
            self._h = hess

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
        q_init = self._q
        x_1 = self.to("cartesian") + np.matmul(self.B_T_inv, value)

        success = False
        rms_s = np.inf
        # NOTE: J. Comput. Chem., 2013, 34, 1842 suggests if step size
        # is larger than 0.5 bohr (= 0.2 Å), internal step can be halved
        # for easier convergence (i.e. damp = 1/2)
        if np.linalg.norm(value) > 0.2:
            damp = 0.5
        else:
            damp = 1.0

        # hybrid SIBT/IBT algorithm
        for i in range(1, MAX_BACK_TRANSFORM_ITERS + 1):
            try:
                x_k = x_k + np.matmul(self.B_T_inv, damp * (s_new - s_k))

                # Rebuild the DIC from back-transformed Cartesians
                q_k = self.primitives.close_to(x_k, q_init)
                s_k = np.matmul(self.U.T, q_k)

                # Rebuild the B matrix every 10 steps
                if i % 10 == 0:
                    self.B = np.matmul(self.U.T, self.primitives.get_B(x_k))
                    self.B_T_inv = np.linalg.pinv(self.B)

                rms_s_old = rms_s
                rms_s = np.sqrt(np.mean(np.square(s_k - s_new)))

                # almost converged, turn off damping
                if rms_s < 1e-6:
                    damp = 1.0
                # RMS going down, reduce damping
                elif rms_s < rms_s_old and i > 1:
                    damp = min(1.2 * damp, 1.0)
                # RMS going up, increase damping
                elif rms_s > rms_s_old:
                    damp = max(0.7 * damp, 0.1)

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
                + f"Final RMS(s) = {rms_s:.8f}"
            )
            x_k = x_1
            if not self.allow_unconverged_back_transform:
                raise CoordinateTransformFailed(
                    "DIC->Cart iterative back-transform did not converge"
                )

        q_k = self.primitives.close_to(x_k, q_init)
        s_k = np.matmul(self.U.T, q_k)
        self.B = np.matmul(self.U.T, self.primitives.get_B(x_k))
        self.B_T_inv = np.linalg.pinv(self.B)

        self[:] = s_k
        self._q = q_k
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

        arr._lambda = np.array([])  # Additional lagrangian multipliers
        return arr

    def __array_finalize__(self, obj: "OptCoordinates") -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        super().__array_finalize__(obj)
        self._lambda = getattr(obj, "_lambda", None)
        return

    @classmethod
    def from_cartesian(
        cls,
        x: "CartesianCoordinates",
        primitives: Optional[PIC] = None,
    ) -> "DICWithConstraints":
        """
        Generate delocalised internal coordinates with constraints
        with the Lagrangian multipliers initialised as zeroes

        Args:
            x: Cartesian coordinates

            primitives: Primitive internal coordinates. If undefined then use
                        all pairwise inverse distances

        Returns:
            (DICWithConstraints): DIC with constraints
        """
        dic = super().from_cartesian(x=x, primitives=primitives)
        dic._lambda = np.zeros(shape=(dic.n_constraints,))
        return dic

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

    def iadd(self, value: np.ndarray) -> "OptCoordinates":
        """
        Add a step in internal coordinates (along with Lagrange multipliers)
        to this set of coordinates, and update the Cartesian coordinates

        Args:
            value: Difference between current and new DICs, and the multipliers
        """
        assert len(value) == len(self) + self.n_constraints
        # separate the coordinates and the lagrange multipliers
        if self.n_constraints > 0:
            delta_lambda = value[-self.n_constraints :]
            self._lambda += delta_lambda
            delta_s = value[: -self.n_constraints]
        else:
            delta_s = value

        return super().iadd(delta_s)

    @property
    def cart_proj_g(self) -> Optional[np.ndarray]:
        """Obtain Cartesian gradient with constraints projected out"""
        if self.g is None:
            return None
        # constrained gradient with inactive terms set to zero
        g_s = self.g
        g_s[self.inactive_indexes] = 0.0
        g_s = g_s[: len(self)]
        # back to Cartesian
        g_x = np.matmul(self.B.T, g_s)
        assert len(g_x) == len(self.to("cart"))
        return g_x

    @property
    def g(self):
        """
        Gradient of the energy, contains the Lagrangian dL/d_λi terms where
        λi is the i-th lagrangian multiplier.
        """
        if self._g is None:
            return None

        n, m = len(self), self.n_constraints
        arr = np.zeros(shape=(n + m,))
        arr[:n] = self._g

        # constrained gradient terms
        for i in range(m):
            arr[n - m + i] -= self._lambda[i] * 1  # λ dC_i/ds_i

        # final dL/dλ_i
        c = self.constrained_primitives
        for i in range(m):
            arr[n + i] = -c[i].delta(self._x)  # C_i(x) = Z - Z_ideal

        return arr

    @g.setter
    def g(self, value):
        """Setting g is not allowed with constraints"""
        raise RuntimeError("Cannot set gradient with constraints enabled")

    @property
    def h(self):
        """
        The Hessian matrix, containing Lagrangian constraint terms

        Returns:
            (np.ndarray):
        """
        if self._h is None:
            return None

        n, m = len(self), self.n_constraints
        arr = np.zeros(shape=(n + m, n + m))

        # Upper left corner is d^2L/ds_i ds_j
        arr[:n, :n] = self._h

        # and the d^2L/ds_i dλ_i = -dC_i/ds_i = -1
        #         d^2L/dλ_i dλ_j = 0
        for i in range(m):
            arr[n + i, :] = arr[:, n + i] = 0.0

        for i in range(m):
            arr[n - m + i, n + i] = arr[n + i, n - m + i] = -1.0

        return arr

    @h.setter
    def h(self, value):
        raise RuntimeError("Cannot set hessian when constraints are enabled")

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
