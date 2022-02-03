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
from typing import Optional
from autode.geom import rotate_columns
from autode.log import logger
from autode.opt.coordinates.primitives import ConstrainedDistance
from autode.opt.coordinates.internals import (PIC,
                                              InverseDistances,
                                              InternalCoordinates)

_max_back_transform_iterations = 100


class DIC(InternalCoordinates):  # lgtm [py/missing-equals]
    """Delocalised internal coordinates"""

    def __repr__(self):
        return f'DIC(n={len(self)})'

    @staticmethod
    def _calc_U(primitives: PIC) -> np.ndarray:
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

        eigvals, eigvecs = np.linalg.eigh(primitives.G)

        # Form a transform matrix from the primitive internals by removing the
        # redundant subspace comprised of small eigenvalues. This forms a set
        # of 3N - 6 non-redundant internals for a system of N atoms
        eigvecs = eigvecs[:, np.where(np.abs(eigvals) > 1E-10)[0]]

        # Move all the weight along constrained distances to the single DIC
        # coordinates, so that Lagrange multipliers are easy to add
        primitive_idxs = [i for i, coord in enumerate(primitives)
                          if isinstance(coord, ConstrainedDistance)]

        if len(primitive_idxs) > 0:
            eigvecs = rotate_columns(eigvecs, *primitive_idxs)

        return eigvecs

    @classmethod
    def from_cartesian(cls,
                       x:         'autode.opt.coordinates.CartesianCoordinates',
                       primitives: Optional[PIC] = None,
                       ) -> 'autode.opt.coordinates.dic.DIC':
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
        logger.info('Converting cartesian coordinates to DIC')
        start_time = time()

        if primitives is None:
            logger.info('Building DICs from all inverse distances')
            primitives = InverseDistances.from_cartesian(x)

        q = primitives(x)
        U = cls._calc_U(primitives)
        dic = cls(input_array=np.matmul(U.T, q))

        dic.U = U                # Transform matrix primitives -> non-redundant

        dic.B = np.matmul(U.T, primitives.B)
        dic.B_T_inv = np.linalg.pinv(dic.B)
        dic._x = x.copy()
        dic.primitives = primitives

        dic.e = x.e                                          # Energy
        dic.update_g_from_cart_g(x.g)                        # Gradient
        dic.update_h_from_cart_h(x.h)                        # and Hessian

        logger.info(f'Transformed in      ...{time() - start_time:.4f} s')
        return dic

    def _update_g_from_cart_g(self,
                              arr: Optional['autode.values.Gradient']
                              ) -> None:
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

    def _update_h_from_cart_h(self,
                              arr: Optional['autode.values.Hessian']
                              ) -> None:
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

    def to(self,
           value: str
           ) -> 'autode.opt.coordinates.base.OptCoordinates':
        """
        Convert these DICs to another type of coordinate

        -----------------------------------------------------------------------
        Arguments:
            value (str): e.g. "Cartesian"

        Returns:
            (autode.opt.coordinates.OptCoordinates): Coordinates
        """

        if value.lower() in ('x', 'cart', 'cartesian'):
            return self._x

        raise ValueError(f'Unknown conversion to {value}')

    def iadd(self,
             value: np.ndarray
             ) -> 'autode.opt.coordidnates.base.OptCoordinates':

        """
        Set some new internal coordinates and update the Cartesian coordinates

        .. math::

            x^(k+1) = x(k) + ({B^T})^{-1}(k)[s_{new} - s(k)]

        for an iteration k.

        ----------------------------------------------------------------------
        Keyword Arguments:

            value (int | float | np.ndarray): Difference between the current
                                              and new DICs. Must be
                                              broadcastable into self.shape.
        Raises:
            (RuntimeError): If the transformation diverges
        """
        start_time = time()
        s_new = self.raw + value

        # Initialise
        s_k, x_k = self.raw, self._x.copy()

        for i in range(1, _max_back_transform_iterations+1):

            x_k = x_k + np.matmul(self.B_T_inv, (s_new - s_k))

            if np.max(np.abs(x_k)) > 1E5:
                raise RuntimeError('Something went very wrong in the back '
                                   'transformation from internal -> carts')

            # Rebuild the primitives & DIC from the back-transformed Cartesians
            s_k = np.matmul(self.U.T, self.primitives(x_k))

            B = np.matmul(self.U.T, self.primitives.B)
            self.B_T_inv = np.linalg.pinv(B)

            if np.average(s_k - s_new)**2 < 1E-16:
                logger.info( f'DIC transformation converged in {i} cycle(s) '
                             f'in {time() - start_time:.4f} s')
                break

            if i == _max_back_transform_iterations:
                raise RuntimeError(f'Failed to transform in '
                                   f'{_max_back_transform_iterations} cycles')

        self[:] = s_k
        self._x = x_k

        return self
