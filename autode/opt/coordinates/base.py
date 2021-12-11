import numpy as np
from typing import Optional, Union, Sequence
from abc import ABC, abstractmethod
from autode.log import logger
from autode.units import (ang, nm, pm, m)
from autode.values import ValueArray, PotentialEnergy


class OptCoordinates(ValueArray, ABC):
    """Coordinates used to perform optimisations"""

    implemented_units = [ang, nm, pm, m]

    @abstractmethod
    def __repr__(self):
        """Representation of these coordinates"""

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

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        for attr in ('units', '_e', '_g', '_h', '_h_inv', 'B', 'B_T_inv'):
            self.__dict__[attr] = getattr(obj, attr, None)

        return None

    @property
    def e(self) -> Optional[PotentialEnergy]:
        """Energy"""
        return self._e

    @e.setter
    def e(self, value):
        """Set the energy"""
        self._e = None if value is None else PotentialEnergy(value)

    @property
    def g(self) -> Optional[np.ndarray]:
        """Gradient of the energy: {dE/dx_i}"""
        return self._g

    @g.setter
    def g(self, value: np.ndarray):
        """Set the gradient of the energy"""
        self._g = value

    @property
    def h(self) -> Optional[np.ndarray]:
        """Second derivatives of the energy: {d^2E/dx_idx_j^2}"""

        if self._h is None and self._h_inv is not None:
            logger.info('Have H^-1 but no H, calculating H')
            self._h = np.linalg.inv(self._h_inv)

        return self._h

    @h.setter
    def h(self, value: np.ndarray):
        """Set the second derivatives of the energy"""
        if not self.h_or_h_inv_has_correct_shape(value):
            raise ValueError(f'Hessian must be an NxN matrix. Had: {value}')

        self._h = value

    @property
    def h_inv(self) -> Optional[np.ndarray]:
        """
        Inverse of the Hessian matrix: :math:`H^{-1}`

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray | None): :math:`H^{-1}`
        """

        if self._h_inv is None and self._h is not None:
            logger.info('Have Hessian but no inverse, so calculating '
                        'explicit inverse')
            self._h_inv = np.linalg.inv(self._h)

        return self._h_inv

    @h_inv.setter
    def h_inv(self, value: np.ndarray):
        """Set the inverse hessian matrix"""
        if not self.h_or_h_inv_has_correct_shape(value):
            raise ValueError('Inverse Hessian must be an NxN matrix. '
                             f'Had: {value}')

        self._h_inv = value

    def h_or_h_inv_has_correct_shape(self, arr: np.ndarray):
        """Does a Hessian or its inverse have the correct shape?"""
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] == len(self)

    @abstractmethod
    def to(self, *args, **kwargs):
        """Transformation between these coordinates and another type"""

    def __setitem__(self, key, value):
        """
        Set an item or slice in these coordinates. Clears the current
        gradient and Hessian as well as clearing setting the coordinates.
        Does NOT check if the current value is close to the current, thus
        the gradient and hessian shouldn't be cleared.
        """

        self.clear_tensors()
        return super().__setitem__(key, value)

    @abstractmethod
    def iadd(self, value: np.ndarray) -> 'OptCoordinates':
        """Inplace addition of some coordinates"""

    def __add__(self, other: Union[np.ndarray, float]):
        """
        Eddition of another set of coordinates. Clears the current
        gradient vector and Hessian matrix.

        Arguments:
            other (np.ndarray): Array to add to the coordinates

        Returns:
            (autode.opt.coordinates.OptCoordinates): Shifted coordinates
        """
        new_coords = self.copy()
        new_coords.clear_tensors()
        new_coords.iadd(other)

        return new_coords

    def __sub__(self, other: Union[np.ndarray, float]):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self.clear_tensors()
        return self.__add__(other)

    def __isub__(self, other):
        self.clear_tensors()
        return self.__sub__(other)

    def clear_tensors(self) -> None:
        """
        Helper function for clearing the energy, gradient and Hessian for these
        coordinates. Called if the coordinates have been perturbed, making
        these quantities not accurate any more for the new coordinates
        """
        self._e, self._g, self._h = None, None, None
        return None
