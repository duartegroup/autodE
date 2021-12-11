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

            s (np.ndarray): Coordinate shift. :math:`s = X_{i+1} - X_i`

            y (np.ndarray): Gradient shift.
                            :math:`y = \nabla E_{i+1} - \nabla E_i`
        """

        self.h = kwargs.get('h', None)
        self.h_inv = kwargs.get('h_inv', None)
        self.s = kwargs.get('s', None)
        self.y = kwargs.get('y', None)

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
            raise RuntimeError('Cannot update H^-1, no inverse defined')

        return self._updated_h_inv

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
            raise RuntimeError('Cannot update H, no Hessian defined')

        return self._updated_h

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


class BFGSUpdate(HessianUpdater):

    @property
    def _updated_h(self) -> np.ndarray:
        """
        Update the Hessian with a BFGS like update

        .. math::

            H_{new} = H + \frac{y y^T}{y^T s} - \frac{H s s^T H}
                                                     {s^T H s}
        """
        h_s = np.matmul(self.h, self.s)

        h_new = (self.h
                 + np.outer(self.y, self.y)/np.dot(self.y, self.s)
                 - (np.outer(h_s, np.matmul(self.s.T, self.h))
                    / np.dot(self.s, h_s)))

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
        """
        logger.info('Calculating H^(-1) with Sherman–Morrison formula')

        s_y = np.dot(self.s, self.y)
        y_h_inv_y = np.dot(self.y, np.matmul(self.h_inv,  self.y))
        s_s = np.outer(self.s, self.s)
        h_inv_y_s = np.matmul(self.h_inv, np.outer(self.y, self.s))
        s_y_h_inv = np.outer(self.s, np.matmul(self.y, self.h_inv))

        h_inv_l = (self.h_inv
                   + (s_y + y_h_inv_y)/(s_y**2) * s_s
                   - (h_inv_y_s + s_y_h_inv)/ s_y)

        return h_inv_l

    @property
    def conditions_met(self) -> bool:
        """BFGS update must meet the secant condition"""

        if np.dot(self.y, self.s) < 0:
            logger.warning('Secant condition not satisfied. Skipping H update')
            return False

        return True


class SR1Update(HessianUpdater):

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
        h_inv_new = (self.h_inv
                     + np.outer(s_h_inv_y, s_h_inv_y)
                       / np.dot(s_h_inv_y, self.y))

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
        r = 1E-8

        if self.h_inv is not None and self.h is None:
            logger.warning('SR1 requires Hessian to determine conditions, '
                           'calculating H from H^(-1)')
            self.h = np.linalg.inv(self.h_inv)

        y_hs = self.y - np.matmul(self.h, self.s)
        s_yhs = np.dot(self.s, y_hs)
        norm_s, norm_yhs = np.linalg.norm(self.s), np.linalg.norm(y_hs)

        return np.abs(s_yhs) > r * norm_s * norm_yhs


class NullUpdate(HessianUpdater):

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
