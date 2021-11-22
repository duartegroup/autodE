"""
Potential energy surface in N-dimensions (distances), enables parallel
calculations over the grid of points, location of saddle points in the
surface and connecting minima and saddle points
"""
import numpy as np
from typing import Dict, Tuple, Union, Optional, Sequence
from autode.values import ValueArray
from autode.units import ha, ev, kcalmol, kjmol, J, ang

# Type is a dictionary keyed with tuples and has a set of floats* as a value
_rs_type = Dict[Tuple[int, int], Union[Tuple, np.ndarray]]


class PESnD:
    """Potential energy surface (PES) in N-dimensions"""

    def __init__(self,
                 species:        Optional['autode.species.species.Species'] = None,
                 rs:             Optional[_rs_type] = None,
                 allow_rounding: bool = True
                 ):
        """
        N-dimensional PES

        -----------------------------------------------------------------------
        Arguments:
            species: Initial species which to evaluate from

            rs: Dictionary of atom indexes (indexed from 0) with associated
                either initial and final, or just final distances. If
                undefined the initial distances are just their current
                values.

            allow_rounding: Allow rounding of a step-size to support an
                            integer number of steps between the initial and
                            final distances
        """
        self._rs = _ListDistances1D(species,
                                    rs_dict=rs if rs is not None else {},
                                    allow_rounding=allow_rounding)

        self._species = species
        self._energies = Energies(np.zeros(self.shape))
        self._coordinates = np.zeros(shape=())

    @property
    def shape(self) -> Tuple:
        """
        Shape of the surface, which is the number of points in each dimension

        -----------------------------------------------------------------------
        Returns:
            (tuple(int)):
        """
        return tuple(len(arr) for arr in self._rs)

    def calculate(self) -> None:
        """
        Calculate the n-dimensional surface
        """

        if self._species is None:
            raise ValueError('Cannot calculate a PES without an initial '
                             'species. Initialise PESNd with a species '
                             'or reactant')

        self._validate_rs()

        # TODO: this function
        return None

    def _validate_rs(self) -> None:
        """
        Ensure that the
        """
        raise NotImplementedError


class _ListDistances1D(list):

    def __init__(self, species, rs_dict, allow_rounding):
        """Construct a list of distance arrays in each dimension"""
        super().__init__([])

        self._species = species
        self._allow_rounding_of_stepsize = allow_rounding

        for idxs, value in rs_dict.items():
            self.append(self._distance1d_from_key_val(idxs, value))

    def _distance1d_from_key_val(self,
                                 atom_idxs: Tuple[int, int],
                                 value:     Union[Tuple, np.ndarray],
                                 ) -> np.ndarray:
        """
        From a 'value' determine the initial and final distances to use

        -----------------------------------------------------------------------
        Arguments:
            value: Some representation of the final and initial points

        Returns:
            (autode.pes.pes_nd.Distances1D):
        """

        if isinstance(value, tuple):
            return self._distance1d_from_key_val_tuple(atom_idxs, value)

        elif isinstance(value, np.ndarray):
            return Distances1D(value, atom_idxs=atom_idxs)

        else:
            raise ValueError('Unable to populate distance array for atom '
                             f'indices {atom_idxs} with: {value}. Must be '
                             f'either a tuple or numpy array')

    def _distance1d_from_key_val_tuple(self,
                                       atom_idxs: Tuple[int, int],
                                       value:     Union[Tuple[float, Union[float, int]],
                                                        Tuple[float, float, Union[float, int]]]
                                       ):
        """
        Determine a array of distances based on a tuple containing either
        a final distance or a number of steps to perform.

        -----------------------------------------------------------------------
        Arguments:
            atom_idxs: Atom indices
            value:

        Returns:
            (autode.pes.pes_nd.Distances1D):
        """

        if len(value) == 2:

            if self._species is None:
                raise ValueError('Cannot determine initial point without '
                                 'a defined species')

            # Have a pair, the final distance and either the number of steps
            # or the step size
            r_init, r_final = self._species.distance(*atom_idxs), value[0]

        elif len(value) == 3:
            # A triple also defines the initial distance
            r_init, r_final = value[0], value[1]

        else:
            raise ValueError(f'Cannot interpret *{value}* as a final '
                             f'distance and number of steps or step size')

        if isinstance(value[-1], int):
            # Integer values must be a number of steps
            num = value[-1]

        elif isinstance(value[-1], float):
            num = int(round(abs((r_final - r_init) / value[-1])))

            if not self._allow_rounding_of_stepsize:
                r_final = (r_init
                           + np.sign(r_final - r_init) * abs(value[-1]) * num)
                num += 1

        else:
            raise ValueError(f'Uninterpretable type: {type(value)}')

        if num <= 1:
            raise ValueError(f'Unsupported number of steps: {num}')

        return Distances1D(np.linspace(r_init, r_final, num=num),
                           atom_idxs=atom_idxs)


class Distances1D(ValueArray):

    implemented_units = [ang]

    def __new__(cls,
                input_array: Union[np.ndarray, Sequence],
                atom_idxs:   Tuple[int, int]):
        """
        Create an array of distances in a single dimension, with associated
        atom indices, indexed from 0

        -----------------------------------------------------------------------
        Arguments:
            input_array: Array of distances e.g. [1.0, 1.1, 1.2] in Å

            atom_idxs: Indices of the atoms involved in this distance
                       e.g. (0, 1)
        """
        arr = super().__new__(cls, input_array=input_array, units=ang)

        if len(atom_idxs) != 2:
            raise ValueError(f'Indices must be a 2-tuple. Had: {atom_idxs}')

        i, j = atom_idxs
        if not (isinstance(i, int) and isinstance(j, int)):
            raise ValueError(f'Atom indices must be integers. Had: {i}, {j}')

        if i < 0 or j < 0:
            raise ValueError(f'Atom indices must be >0: Had {i}, {j}')

        arr.atom_idxs = atom_idxs
        return arr

    @property
    def min(self) -> float:
        """
        Minimum value of the array

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        return min(self)

    @property
    def max(self) -> float:
        """
        Maximum value of the array

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        return max(self)

    def __repr__(self):
        return f'Distances(n={len(self)}, [{self.min, self.max}])'


class Energies(ValueArray):

    implemented_units = [ha, ev, kcalmol, kjmol, J]

    def __repr__(self):
        """Representation of the energies in a PES"""
        return f'PES{self.ndim}d'
