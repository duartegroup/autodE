import numpy as np
from abc import ABC, abstractmethod


class Primitive(ABC):
    """Primitive internal coordinate"""

    @abstractmethod
    def __call__(self, x: 'autode.opt.CartesianCoordinates'):
        """Return the value of this PIC given a set of cartesian coordinates"""

    @abstractmethod
    def derivative(self,
                   i: int,
                   k: str,
                   x: 'autode.opt.CartesianCoordinates'):
        """
        Calculate the derivative with respect to a cartesian coordinate::

                dq   |
            ---------|
            dx_(i, k)|_x=x0

        ----------------------------------------------------------------------
        Argument:
            i (int): Cartesian index to take the derivative with respect to;
                     0-N for N atoms

            k (str): Cartesian component (x, y, z) to take the derivative with
                      respect to. {'x', 'y', 'z'}

            x (autode.opt.cartesian.CartesianCoordinates): Cartesian coordinates

        Returns:
            (float): Derivative
        """


class InverseDistance(Primitive):

    def __init__(self, idx_i, idx_j):
        """
         1 / |x_i - x_j|        for a set of cartesian coordinates x

        Arguments:
            idx_i (int): Atom index
            idx_j (int): Atom index
        """

        self.idx_i = int(idx_i)
        self.idx_j = int(idx_j)

    def derivative(self, i, component, x):
        """See ABC for docstring"""

        _x = x.reshape((-1, 3))
        k = {'x': 0, 'y': 1, 'z': 2}[component]

        if i != self.idx_i and i != self.idx_j:
            return 0                 # Atom does not form part of this distance

        elif i == self.idx_i:
            return - (_x[i, k] - _x[self.idx_j, k]) * self(x)**3

        else:  # i == self.idx_j:
            return (_x[self.idx_i, k] - _x[self.idx_j, k]) * self(x)**3

    def __call__(self, x):
        """1 / |x_i - x_j| """
        _x = x.reshape((-1, 3))
        return 1.0 / np.linalg.norm(_x[self.idx_i] - _x[self.idx_j])
