import numpy as np
from abc import ABC, abstractmethod


class Primitive(ABC):
    """Primitive internal coordinate"""

    @abstractmethod
    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates'
                 ) -> float:
        """Return the value of this PIC given a set of cartesian coordinates"""

    @abstractmethod
    def derivative(self,
                   i: int,
                   k: str,
                   x: 'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        r"""
        Calculate the derivative with respect to a cartesian coordinate

        .. math::

            \frac{dq}
                  {d\boldsymbol{X}_{i, k}} {\Bigg\rvert}_{X=X0}

        where :math:`q` is the primitive coordinate and :math:`\boldsymbol{X}`
        are the cartesian coordinates.

        ----------------------------------------------------------------------
        Arguments:
            i: Cartesian index to take the derivative with respect to;
                0-N for N atoms

            k: Cartesian component (x, y, z) to take the derivative with
               respect to. {'x', 'y', 'z'}

            x: Cartesian coordinates

        Returns:
            (float): Derivative
        """


class InverseDistance(Primitive):

    def __init__(self,
                 idx_i: int,
                 idx_j: int):
        r"""
        Inverse distance between a pair of atoms

        .. math::

            q = \frac{1}
                    {|\boldsymbol{X}_i - \boldsymbol{X}_j|}


        for a set of cartesian coordinates :math:`\boldsymbol{X}`.

        Arguments:
            idx_i: Atom index
            idx_j: Atom index
        """

        self.idx_i = int(idx_i)
        self.idx_j = int(idx_j)

    def derivative(self,
                   i:         int,
                   component: str,
                   x:        'autode.opt.coordinates.CartesianCoordinates'
                   ) -> float:
        """
        Derivative

        -----------------------------------------------------------------------
        See Also:
            :py:meth:`Primitive.derivative <Primitive.derivative>`
        """

        _x = x.reshape((-1, 3))
        k = {'x': 0, 'y': 1, 'z': 2}[component]

        if i != self.idx_i and i != self.idx_j:
            return 0                 # Atom does not form part of this distance

        elif i == self.idx_i:
            return - (_x[i, k] - _x[self.idx_j, k]) * self(x)**3

        else:  # i == self.idx_j:
            return (_x[self.idx_i, k] - _x[self.idx_j, k]) * self(x)**3

    def __call__(self,
                 x: 'autode.opt.coordinates.CartesianCoordinates'
                 ) -> float:
        """1 / |x_i - x_j| """
        _x = x.reshape((-1, 3))
        return 1.0 / np.linalg.norm(_x[self.idx_i] - _x[self.idx_j])
