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
|
"""
import numpy as np
from time import time
from typing import Type, Optional
from autode.log import logger
from autode.opt.coordinates.internals import (PIC,
                                              InverseDistances,
                                              InternalCoordinates)


class DIC(InternalCoordinates):
    """Delocalised internal coordinates"""

    def __repr__(self):
        return f'DIC(n={len(self)})'

    @staticmethod
    def U(primitives: PIC) -> np.ndarray:
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

        B_q = primitives.B
        G = np.matmul(B_q, B_q.T)

        w, v = np.linalg.eigh(G)  # Eigenvalues and eigenvectors respectively

        # Form a transform matrix from the primitive internals to a set of
        # 3N - 6 non-redundant internals, s
        return v[:, np.where(np.abs(w) > 1E-10)[0]]

    @classmethod
    def from_cartesian(cls,
                       x:             'autode.opt.cartesian.CartesianCoordinates',
                       primitive_type: Type[PIC] = InverseDistances
                       ) -> 'autode.opt.coordinates.dic.DIC':
        """
        Convert cartesian coordinates to primitives then to delocalised
        internal coordinates (DICs), of which there should be 3N-6 for a
        polyatomic system with N atoms

        -----------------------------------------------------------------------
        Arguments:
            x (autode.opt.CartesianCoordinates): Cartesian coordinates

            primitive_type (autode.opt.internals.PIC): Primitive internal
                           coordinates, constructable from Cartesian

        Returns:
            (autode.opt.coordinates.DIC): Delocalised internal coordinates
        """
        logger.info('Converting cartesian coordinates to DIC')
        start_time = time()

        primitives = primitive_type(x)
        U = cls.U(primitives)

        s = cls(input_array=np.matmul(U.T, primitives.q))
        s.e = x.e  # Energy

        s.B = np.matmul(U.T, primitives.B)
        s.B_T_inv = np.linalg.pinv(s.B)
        s._x = x.copy()
        s.primitive_type = primitive_type

        s.update_g_from_cart_g(x.g)

        # and Hessian
        if x.h is not None:
            # NOTE: This is not the full transformation as noted in
            # 10.1063/1.471864 only an approximate Hessian is required(?)
            s.h = np.linalg.multi_dot((s.B_T_inv.T, x.h, s.B_T_inv))

        logger.info(f'Transformed in      ...{time() - start_time:.4f} s')
        return s

    def update_g_from_cart_g(self,
                             arr: Optional['autode.values.Gradient']
                             ) -> None:
        """
        Updates the gradient from a calculated Cartesian gradient

        -----------------------------------------------------------------------
        Arguments:
            arr: Gradient array
        """
        if arr is None:
            self._x.g, self.g = None, None

        else:
            self._x.g = arr.flatten()
            self.g = np.matmul(self.B_T_inv.T, self._x.g)

        return None

    def to(self, value: str) -> 'autode.opt.coordinates.base.OptCoordinates':
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

    def update(self, delta) -> None:
        """
        Set some new internal coordinates and update the Cartesian coordinates

        .. math::

            x^(k+1) = x(k) + ({B^T})^{-1}(k)[s_{new} - s(k)]

        for an iteration k.

        ----------------------------------------------------------------------
        Keyword Arguments:

            delta (int | float | np.ndarray): Difference between the current
                                              and new DICs. Must be
                                              broadcastable into self.shape.
        Raises:
            (RuntimeError): If the transformation diverges
        """
        start_time = time()
        s_new = self.raw + delta

        # Initialise
        s_k, x_k = self.raw, self._x.copy()
        U = self.U(primitives=self.primitive_type(x_k))

        iteration = 0

        # Converge to an RMS difference of less than a tolerance
        while np.average(s_k - s_new) ** 2 > 1E-16 and iteration < 100:

            x_k = x_k + np.matmul(self.B_T_inv, (s_new - s_k))

            if np.max(np.abs(x_k)) > 1E5:
                raise RuntimeError('Something went very wrong in the back '
                                   'transformation from internal -> carts')

            # Rebuild the primitives from the back-transformed Cartesians
            primitives = self.primitive_type(x_k)
            s_k = np.matmul(U.T, primitives.q)

            B = np.matmul(U.T, primitives.B)
            self.B_T_inv = np.linalg.pinv(B)

            iteration += 1

        logger.info(f'DIC transformation converged in {iteration} cycles and '
                    f'{time() - start_time:.4f} s')

        self[:] = s_k
        self.clear_tensors()

        self._x = x_k
        self._x.clear_tensors()
        return None

    def iadd(self,
             value: np.ndarray) -> 'autode.opt.coordidnates.base.OptCoordinates':
        """Inplace addition of another set of coordinates"""
        self.update(delta=value)
        return self
