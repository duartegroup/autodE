import numpy as np

from abc import ABC, abstractmethod
from autode.log import logger


class HessianUpdater(ABC):
    """Update strategy for the (inverse) Hessian matrix"""

    def __init__(self, **kwargs):
        r"""
        Hessian updater

        ----------------------------------------------------------------------
        Keyword Arguments:
            h (np.ndarray): Hessian (:math:`H`), shape = (N, N)

            h_inv (np.ndarray): Inverse Hessian (:math:`H^{-1}`), shape = (N, N)

            s (np.ndarray): Coordinate shift. :math:`s = R_{i+1} - R_i`

            y (np.ndarray): Gradient shift.
                            :math:`y = \nabla E_{i+1} - \nabla E_i`

            subspace_idxs (list(int)): Indexes of the components of the
                                       hessian to update
        """

        self.h = kwargs.get("h", None)
        self.h_inv = kwargs.get("h_inv", None)
        self._h_init, self._h_inv_init = None, None

        self.s = kwargs.get("s", None)
        self.y = kwargs.get("y", None)
        self.subspace_idxs = kwargs.get("subspace_idxs", None)
        self._apply_subspace()

    def _apply_subspace(self) -> None:
        """
        Reduce the step, gradient difference vectors and the Hessian & inverse
        to include only a subset of the total elements
        """
        idxs = self.subspace_idxs

        if idxs is None:
            return  # Cannot apply with no defined indexes

        if len(idxs) == 0:
            raise ValueError(
                "Cannot reduce s, y, h to 0 dimensional. "
                "idxs must have at least one element"
            )

        logger.info(f"Updated hessian will have shape {len(idxs)}x{len(idxs)}")

        for attr in ("h", "h_inv"):
            m = getattr(self, attr)
            setattr(self, f"_{attr}_init", None if m is None else m.copy())
            setattr(self, attr, None if m is None else m[:, idxs][idxs, :])

        for attr in ("s", "y"):
            v = getattr(self, attr)
            setattr(self, attr, None if v is None else v[idxs])

        return None

    def _matrix_in_full_space(
        self, m: np.ndarray, m_sub: np.ndarray
    ) -> np.ndarray:
        """
        Create a Hessian in the full initial space i.e. having only updated
        the components present in self.subspace_idxs. Also ensures that the
        Hessian is Hermitian
        """
        assert self.subspace_idxs is not None

        for i, idx_i in enumerate(self.subspace_idxs):
            for j, idx_j in enumerate(self.subspace_idxs):
                m[idx_i, idx_j] = m_sub[i, j]

        return _ensure_hermitian(m)

    @property
    def updated_h_inv(self) -> np.ndarray:
        """
        Calculate H^{-1} from a previous inverse Hessian, coordinate shift and
        gradient  shift

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): :math:`H^{-1}`

        Raises:
            (RuntimeError): If the update fails
        """

        if self.h_inv is None:
            raise RuntimeError("Cannot update H^-1, no inverse defined")

        if self._h_inv_init is None:
            return self._updated_h_inv

        return self._matrix_in_full_space(
            self._h_inv_init, self._updated_h_inv
        )

    @property
    def updated_h(self) -> np.ndarray:
        """
        Calculate H from a previous Hessian, coordinate shift and gradient
        shift

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): :math:`H`

        Raises:
            (RuntimeError): If the update fails
        """

        if self.h is None:
            raise RuntimeError("Cannot update H, no Hessian defined")

        if self._h_init is None:
            return self._updated_h

        return self._matrix_in_full_space(self._h_init, self._updated_h)

    @property
    @abstractmethod
    def conditions_met(self) -> bool:
        """Are the conditions met to update the Hessian with this method?"""

    @property
    @abstractmethod
    def _updated_h(self) -> np.ndarray:
        """Calculate H"""

    @property
    @abstractmethod
    def _updated_h_inv(self) -> np.ndarray:
        """Calculate H^{-1}"""

    def __str__(self):
        return self.__repr__()


class BFGSUpdate(HessianUpdater):
    def __repr__(self):
        return "BFGS"

    @property
    def _updated_h(self) -> np.ndarray:
        r"""
        Update the Hessian with a BFGS like update

        .. math::

            H_{new} = H + \frac{y y^T}{y^T s} - \frac{H s s^T H}
                                                     {s^T H s}


        -----------------------------------------------------------------------
        See Also:
            :py:meth:`BFGSUpdate._updated_h_inv <BFGSUpdate._updated_h_inv>`
        """
        h_s = np.matmul(self.h, self.s)

        h_new = (
            self.h
            + np.outer(self.y, self.y) / np.dot(self.y, self.s)
            - (
                np.outer(h_s, np.matmul(self.s.T, self.h))
                / np.dot(self.s, h_s)
            )
        )

        return h_new

    @property
    def _updated_h_inv(self):
        r"""
        Sherman–Morrison inverse matrix update

        .. math::

            H_{new}^{-1} = H^{-1} +
                            \frac{(s^Ty + y^T H^{-1} y) s^T s}
                                 {s^T y} -
                            \frac{H^{-1} y s^T + s y^T H^{-1}}
                                 {s^T y}

        where :math:`s = x_{l} - x_{l-1},\; \boldsymbol{y} =
        \nabla E_l - \nabla E_{l-1}`.

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`BFGSUpdate._updated_h <BFGSUpdate._updated_h>`
        """
        logger.info("Calculating H^(-1) with Sherman–Morrison formula")

        s_y = np.dot(self.s, self.y)
        y_h_inv_y = np.dot(self.y, np.matmul(self.h_inv, self.y))
        s_s = np.outer(self.s, self.s)
        h_inv_y_s = np.matmul(self.h_inv, np.outer(self.y, self.s))
        s_y_h_inv = np.outer(self.s, np.matmul(self.y, self.h_inv))

        h_inv_l = (
            self.h_inv
            + (s_y + y_h_inv_y) / (s_y**2) * s_s
            - (h_inv_y_s + s_y_h_inv) / s_y
        )

        return h_inv_l

    @property
    def conditions_met(self) -> bool:
        """BFGS update must meet the secant condition"""

        if np.dot(self.y, self.s) < 0:
            logger.warning("Secant condition not satisfied. Skipping H update")
            return False

        return True


class BFGSPDUpdate(BFGSUpdate):
    """BFGS update while ensuring positive definiteness"""

    def __init__(self, min_eigenvalue: float = 1e-5, **kwargs):
        super().__init__(**kwargs)

        self.min_eigenvalue = min_eigenvalue

    def __repr__(self):
        return "BFGS positive definite"

    @property
    def conditions_met(self) -> bool:
        """Are all the conditions met to update the Hessian"""

        eigvals = np.linalg.eigvals(self._updated_h)
        return super().conditions_met and np.all(eigvals > self.min_eigenvalue)


class BFGSDampedUpdate(BFGSPDUpdate):
    """
    Powell damped BFGS update that ensures reasonable conditioning with the
    'positive definite' conditions still imposed
    """

    @property
    def _updated_h(self) -> np.ndarray:
        """
        Powell damped BFGS from: Math. Prog. Comp. (2016) 8:435–459
        (10.1007/s12532-016-0101-2)
        """

        h, s, y = self.h, self.s, self.y
        shs = np.linalg.multi_dot((s.T, h, s))

        if s.dot(y) < 0.2 * shs:
            theta = (0.8 * shs) / (shs - s.dot(y))
        else:
            theta = 1.0

        y_ = theta * y - (1.0 - theta) * h.dot(s)

        h_new = (
            h
            - (np.outer(h.dot(s), np.matmul(s.T, h)) / shs)
            + np.outer(y_, y_) / np.dot(y_, s)
        )

        return h_new


class SR1Update(HessianUpdater):
    def __repr__(self):
        return "SR1"

    @property
    def _updated_h(self) -> np.ndarray:
        r"""
        Update H using a symmetric-rank 1 (SR1) update

        .. math::
            H_{new} = H + \frac{(y- Hs)(y - Hs)^T}
                               {(y- Hs)^T s}
        """

        y_hs = self.y - np.matmul(self.h, self.s)
        h_new = self.h + np.outer(y_hs, y_hs) / np.dot(y_hs, self.s)

        return h_new

    @property
    def _updated_h_inv(self) -> np.ndarray:
        r"""
        Update H^-1 using a symmetric-rank 1 (SR1) update

        .. math::

            H_{new}^{-1} = H^{-1} + \frac{(s- H^{-1}y)(s - H^{-1}y)^T}
                                          {(s- Hy)^T y}

        """

        s_h_inv_y = self.s - np.matmul(self.h_inv, self.y)
        h_inv_new = self.h_inv + (
            np.outer(s_h_inv_y, s_h_inv_y) / np.dot(s_h_inv_y, self.y)
        )

        return h_inv_new

    @property
    def conditions_met(self) -> bool:
        r"""
        Condition for SR1 update. See:
        https://en.wikipedia.org/wiki/Symmetric_rank-one

        .. math::

            |s (y - Hs)| \ge r ||s|| \cdot ||y - Hs||

        where :math:`r \in (0, 1)` = 1E-8.
        """
        r = 1e-8

        if self.h_inv is not None and self.h is None:
            logger.warning(
                "SR1 requires Hessian to determine conditions, "
                "calculating H from H^(-1)"
            )
            self.h = np.linalg.inv(self.h_inv)

        y_hs = self.y - np.matmul(self.h, self.s)
        s_yhs = np.dot(self.s, y_hs)
        norm_s, norm_yhs = np.linalg.norm(self.s), np.linalg.norm(y_hs)

        return np.abs(s_yhs) > r * norm_s * norm_yhs


class NullUpdate(HessianUpdater):
    def __repr__(self):
        return "Null"

    @property
    def conditions_met(self) -> bool:
        """Conditions are always met for a null optimiser"""
        return True

    @property
    def _updated_h(self) -> np.ndarray:
        """Updated H is just the input Hessian"""
        return self.h.copy()

    @property
    def _updated_h_inv(self) -> np.ndarray:
        """Updated inverse H is just the input inverse Hessian"""
        return self.h_inv.copy()


class BofillUpdate(HessianUpdater):
    """
    Hessian update strategy suggested by Bofill[2] with notation taken from
    ref. [1].

    [1] V. Bakken, T. Helgaker, JCP, 117, 9160, 2002
    [2] J. M. Bofill, J. Comput. Chem., 15, 1, 1994
    """

    # Threshold on |Δg - HΔx| below which the Hessian will not be updated, to
    # prevent dividing by zero
    min_update_tol = 1e-6

    def __repr__(self):
        return "Bofill"

    @property
    def _updated_h(self) -> np.ndarray:
        r"""
        Bofill Hessian update, interpolating between MS and PBS update
        strategies. Follows ref. [1] where the notation is

        .. math::
            h = \boldsymbol{G}_{i-1}

            y = \Delta\boldsymbol{g} = \boldsymbol{g}_i - \boldsymbol{g}_{i-1}

            s = \Delta\boldsymbol{x} = \boldsymbol{x}_i - \boldsymbol{x}_{i-1}

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): H
        """
        logger.info("Updating the Hessian with the Bofill scheme")

        # from ref. [1] the approximate Hessian (G) is self.H
        G_i_1 = self.h  # G_{i-1}
        dE_i = self.y - np.dot(G_i_1, self.s)  # ΔE_i = Δg_i - G_{i-1}Δx_i

        if np.linalg.norm(dE_i) < self.min_update_tol:
            logger.warning(
                f"|Δg_i - G_i-1Δx_i| < {self.min_update_tol:.4f} "
                f"not updating the Hessian"
            )
            return self.h.copy()

        # G_i^MS eqn. 42 from ref. [1]
        G_i_MS = G_i_1 + np.outer(dE_i, dE_i) / np.dot(dE_i, self.s)

        # G_i^PBS eqn. 43 from ref. [1]
        dxTdg = np.dot(self.s, self.y)
        G_i_PSB = (
            G_i_1
            + (
                (np.outer(dE_i, self.s) + np.outer(self.s, dE_i))
                / np.dot(self.s, self.s)
            )
            - (
                (
                    (dxTdg - np.linalg.multi_dot((self.s, G_i_1, self.s)))
                    * np.outer(self.s, self.s)
                )
                / np.dot(self.s, self.s) ** 2
            )
        )

        # ϕ from eqn. 46 from ref [1]
        phi_bofill = 1.0 - (
            np.dot(self.s, dE_i) ** 2
            / (np.dot(self.s, self.s) * np.dot(dE_i, dE_i))
        )

        logger.info(f"ϕ_Bofill = {phi_bofill:.6f}")

        return (1.0 - phi_bofill) * G_i_MS + phi_bofill * G_i_PSB

    @property
    def _updated_h_inv(self) -> np.ndarray:
        """Updated inverse Hessian is available only from the updated H"""
        return np.linalg.inv(self._updated_h)

    @property
    def conditions_met(self) -> bool:
        """
        No conditions are need to be satisfied to perform a Bofill update,
        apart from that on the shapes of the vectors
        """
        return True


class FlowchartUpdate(HessianUpdater):
    """
    A hybrid update scheme combining BFGS, SR1 and PSB Hessian update
    formulae. Proposed in  A. B. Birkholz and H. B. Schlegel in
    Theor. Chem. Acc., 135 (84), 2016. This implementation is slightly
    modified.
    """

    def __repr__(self):
        return "Flowchart"

    @property
    def _updated_h(self) -> np.ndarray:
        """
        Flowchart (or FlowPSB) Hessian update scheme, that dynamically
        switches between BFGS and SR1 depending on some criteria.

        Alternatively switches to PSB update as a fallback if none of
        the criteria are satisfied. Notation follows A. B. Birkholz,
        H. B. Schlegel, Theor. Chem. Acc., 135 (84), 2016.

        Returns:
            (np.ndarray): H
        """
        z = self.y - np.matmul(self.h, self.s)
        sr1_criteria = np.dot(z, self.s) / (
            np.linalg.norm(z) * np.linalg.norm(self.s)
        )
        bfgs_criteria = np.dot(self.y, self.s) / (
            np.linalg.norm(self.y) * np.linalg.norm(self.s)
        )
        if sr1_criteria < -0.1:
            h_new = self.h + np.outer(z, z) / np.dot(z, self.s)
            return h_new
        elif bfgs_criteria > 0.1:
            bfgs_delta_h = np.outer(self.y, self.y) / np.dot(self.y, self.s)
            bfgs_delta_h -= np.linalg.multi_dot(
                (self.h, self.s.reshape(-1, 1), self.s.reshape(1, -1), self.h)
            ) / np.linalg.multi_dot(
                (self.s.flatten(), self.h, self.s.flatten())
            )
            h_new = self.h + bfgs_delta_h
            return h_new
        else:
            # Notation copied from Bofill update
            G_i_1 = self.h  # G_{i-1}
            dE_i = self.y - np.dot(G_i_1, self.s)  # ΔE_i = Δg_i - G_{i-1}Δx_i
            dxTdg = np.dot(self.s, self.y)
            G_i_PSB = (
                G_i_1
                + (
                    (np.outer(dE_i, self.s) + np.outer(self.s, dE_i))
                    / np.dot(self.s, self.s)
                )
                - (
                    (
                        (dxTdg - np.linalg.multi_dot((self.s, G_i_1, self.s)))
                        * np.outer(self.s, self.s)
                    )
                    / np.dot(self.s, self.s) ** 2
                )
            )
            return G_i_PSB

    @property
    def _updated_h_inv(self) -> np.ndarray:
        """Flowchart update is only available for Hessian"""
        return np.linalg.inv(self._updated_h)

    @property
    def conditions_met(self) -> bool:
        """
        Flowchart update does not have any conditions, as
        update scheme is dynamically selected
        """
        return True


class BFGSSR1Update(HessianUpdater):
    """
    Interpolates between BFGS and SR1 update in a fashion similar
    to Bofill updates, but suited for minimisations. Proposed by
    Farkas and Schlegel in J. Chem. Phys., 111, 1999, 10806
    """

    def __repr__(self):
        return "BFGS-SR1"

    @property
    def _updated_h(self) -> np.ndarray:
        """
        Hybrid BFGS and SR1 update. The mixing parameter phi is defined
        as the square root of the (1 - phi_Bofill) used in Bofill update.

        Returns:
            (np.ndarray): The updated hessian
        """
        bfgs_delta_h = np.outer(self.y, self.y) / np.dot(self.y, self.s)
        bfgs_delta_h -= np.linalg.multi_dot(
            (self.h, self.s.reshape(-1, 1), self.s.reshape(1, -1), self.h)
        ) / np.linalg.multi_dot((self.s.flatten(), self.h, self.s.flatten()))

        y_hs = self.y - np.matmul(self.h, self.s)
        sr1_delta_h = np.outer(y_hs, y_hs) / np.dot(y_hs, self.s)

        # definition according to Farkas, Schlegel, J Chem. Phys., 111, 1999
        # NOTE: this phi is (1 - original_phi_bofill)
        phi = np.dot(self.s, y_hs) ** 2 / (
            np.dot(self.s, self.s) * np.dot(y_hs, y_hs)
        )
        sqrt_phi = np.sqrt(phi)
        logger.info(f"BFGS-SR1 update: ϕ = {sqrt_phi:.4f}")
        return self.h + sqrt_phi * sr1_delta_h + (1 - sqrt_phi) * bfgs_delta_h

    @property
    def _updated_h_inv(self) -> np.ndarray:
        """For BFGS-SR1 update, only hessian is available"""
        return np.linalg.inv(self._updated_h)

    @property
    def conditions_met(self) -> bool:
        """No conditions need to be satisfied for BFGS-SR1 update"""
        return True


def _ensure_hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0
