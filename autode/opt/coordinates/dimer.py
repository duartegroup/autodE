import numpy as np
from typing import Union, Sequence
from autode.opt.coordinates.cartesian import CartesianCoordinates


class DimerCoordinates(CartesianCoordinates):

    def __new__(cls,
                input_array: Union[Sequence, np.ndarray],
                units:       Union[str, 'autode.units.Unit'] = 'Å'
                ) -> 'OptCoordinates':
        """New instance of these coordinates"""

        arr = super().__new__(cls, np.array(input_array), units)

        arr._e = None    # Energy

        """
        Compared to standard Cartesian coordinates these arrays have and
        additional dimension, one for translation (subscript T) and one
        for rotation. i.e _g is a 2x3N_atoms matrix and 
        _h a 2x3N_atomsx3N_atoms tensor
        """
        arr._g = None    # Gradient: {dE/dX_T, dE/dx_R}
        arr._h = None    # Hessian: {d2E/dXdY_T, d2E/dXdY_R}

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        for attr in ('units', '_e', '_g', '_h'):
            self.__dict__[attr] = getattr(obj, attr, None)

        return None

    @classmethod
    def from_species(cls,
                     species1: 'autode.species.Species',
                     species2: 'autode.species.Species'
                     ) -> 'DimerCoordinates':
        """
        Initialise a set of DimerCoordinates from two species, i.e. those
        either side of the saddle point.
        """
        if not species1.has_identical_composition_as(species2):
            raise ValueError('Cannot form a set of dimer coordinates from two '
                             'species with a different number of atoms')

        coords_pair = (species1.coordinates, species2.coordinates)
        coords = cls(np.concatenate(coords_pair, axis=None))

        return coords
