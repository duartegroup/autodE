import numpy as np
from typing import Union, Sequence
from autode.opt.coordinates.cartesian import CartesianCoordinates


class DimerCoordinates(CartesianCoordinates):

    def __new__(cls,
                input_array: Union[Sequence, np.ndarray],
                units:       Union[str, 'autode.units.Unit']
                ) -> 'OptCoordinates':
        """New instance of these coordinates"""

        arr = super().__new__(cls, np.array(input_array), units)

        arr._e = None              # Energy
        arr._g = None              # Gradient: dE/dX
        arr._h = None              # Hessian:  d2E/dX_idX_j
        arr._h_inv = None          # Inverse of the Hessian: H^-1
        arr.B = None               # Wilson B matrix
        arr.B_T_inv = None         # Generalised inverse of B
        arr.U = np.eye(len(arr))   # Transform matrix

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        for attr in ('units', '_e', '_g', '_h', '_h_inv', 'U', 'B', 'B_T_inv'):
            self.__dict__[attr] = getattr(obj, attr, None)

        return None


