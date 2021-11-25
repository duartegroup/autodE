"""
Potential energy surface in N-dimensions (distances), enables parallel
calculations over the grid of points, location of saddle points in the
surface and connecting minima and saddle points
"""
import numpy as np
import itertools as it
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Sequence
from autode.config import Config
from autode.log import logger
from autode.values import ValueArray, Energy
from autode.units import ha, ev, kcalmol, kjmol, J, ang


# Type is a dictionary keyed with tuples and has a set of floats* as a value
_rs_type = Dict[Tuple[int, int], Union[Tuple, np.ndarray]]


class PESnD(ABC):
    """Potential energy surface (PES) in N-dimensions"""

    def __init__(self,
                 species:        Optional['autode.species.species.Species'] = None,
                 rs:             Optional[_rs_type] = None,
                 allow_rounding: bool = True
                 ):
        """
        N-dimensional PES

        # TODO: Add example initialisation

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

        # Dynamically add public attributes for r1, r2, ... etc. as nD arrays
        for i, meshed_rs in enumerate(np.meshgrid(*self._rs, indexing='ij')):
            setattr(self, f'r{i+1}', meshed_rs)

        self._species = species
        self._energies = Energies(np.zeros(self.shape), units='Ha')
        self._coordinates = np.zeros(shape=(*self.shape, ))

        # Attributes set when calling calcualte()
        self._method:   Optional['autode.wrappers.base.Method'] = None
        self._n_cores:  Optional[int] = None
        self._keywords: Optional['autode.wrappers.keywords.Keywords'] = None

    @property
    def shape(self) -> Tuple:
        """
        Shape of the surface, which is the number of points in each dimension

        -----------------------------------------------------------------------
        Returns:
            (tuple(int)):
        """
        return tuple(len(arr) for arr in self._rs)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in this PES

        -----------------------------------------------------------------------
        Returns:
            (int):
        """
        return len(self._rs)

    @property
    def origin(self) -> Tuple:
        """
        Tuple of the origin e.g. (0,) in 1D and (0, 0, 0) in 3D

        -----------------------------------------------------------------------
        Returns:
            (tuple(int, ...)):
        """
        return tuple(0 for _ in range(self.ndim))

    def calculate(self,
                  method:   'autode.wrapper.ElectronicStructureMethod',
                  keywords:  Optional['autode.wrappers.Keywords'] = None,
                  n_cores:   Optional[int] = None
                  ) -> None:
        """
        Calculate the surface

        -----------------------------------------------------------------------
        Arguments:
            method: Method to use

            keywords: Keywords to use. If None then will use method.keywords.sp
                      for an unrelaxed or method.keywords.opt for a relaxed

            n_cores: Number of cores. If None then use ade.Config.n_cores
        """
        if self._species is None:
            raise ValueError('Cannot calculate a PES without an initial '
                             'species. Initialise PESNd with a species '
                             'or reactant')

        if keywords is None:
            self._keywords = self._default_keywords(method)
            logger.info('PES calculation keywords not specified, using:\n'
                        f'{self._keywords}')
        else:
            self._keywords = keywords

        # Coordinates tensor is the shape of the PES plus (N, 3) dimensions
        self._coordinates = np.zeros(shape=(*self.shape, self._species.n_atoms, 3),
                                     dtype=np.float64)

        # Set the coordinates of the first point in the PES, and other attrs
        self._coordinates[self.origin] = self._species.coordinates
        self._method = method
        self._n_cores = Config.n_cores if n_cores is None else n_cores

        self._calculate()
        return None

    @abstractmethod
    def _default_keywords(self,
                          method: 'autode.wrapper.ElectronicStructureMethod'
                          ) -> 'autode.wrappers.Keywords':
        """
        Default keywords to use for this type of PES e.g. opt or sp

        -----------------------------------------------------------------------
        Arguments:
            method:

        Returns:
            (autode.wrappers.keywords.Keywords):
        """

    @abstractmethod
    def _calculate(self) -> None:
        """Calculate the surface, using method, keywords, n_cores attributes"""

    def _points(self) -> Sequence[Tuple]:
        """
        A list of points in this PES sorted by their sum. For example, for a
        1D PES containing 3 points the list is: [(0,), (1,), (2,)] while for
        a 2D PES of 4 total points the points list is:
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        used for enumerating over the surface in from the initial species
        point (located at the origin).

        -----------------------------------------------------------------------
        Returns:
            (list(tuple(int, ..))): List of points
        """
        ranges = (range(len(r)) for r in self._rs)

        return sorted(it.product(*ranges), key=lambda x: sum(x))

    def _point_name(self, point: Tuple) -> str:
        """
        Name of a particular point in the surface

        -----------------------------------------------------------------------
        Arguments:
            point: Indices of the point

        Returns:
            (str):
        """
        return f'{self._species.name}_scan_{"-".join([str(p) for p in point])}'

    def _point_is_contained(self, point: Tuple) -> bool:
        """
        Is a point contained on this PES, defined by its indices. For example,
        (-1,) is never on a 1D PES, (2,) is on a 1D PES with 3 points in it
        and (1,) is not on a 2D PES as it doesn't have the same dimension

        -----------------------------------------------------------------------
        Arguments:
            point: Indices of a point on the grid

        Returns:
            (bool): If the point is on the PES
        """
        if len(point) != self.ndim:
            return False

        if sum(point) < 0:
            return False

        if any(p_n >= s_n or p_n < 0 for p_n, s_n in zip(point, self.shape)):
            return False

        return True

    def _point_has_energy(self, point: Tuple) -> bool:
        """
        Does a point have a defined energy? Energies are initialised to
        zero so only need to check that the energy is not vanishing

        -----------------------------------------------------------------------
        Arguments:
            point:

        Returns:
            (bool):

        Raises:
            (IndexError): If the point is not on the PES
        """
        return not np.isclose(self._energies[point], 0.0, atol=1E-10)

    def __getitem__(self,
                    indices: Union[Tuple, int]):
        """
        Get a value on this potential energy surface (PES) at a (set of)
        indices

        -----------------------------------------------------------------------
        Arguments:
            indices:

        Returns:
            (autode.values.Energy): Energy
        """
        return Energy(self._energies[indices], units=self._energies.units)

    def __repr__(self):
        return f'PES(shape={self.shape})'


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

        Raises:
            (ValueError): If the value is not of the correct type
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
            num = int(round(abs((r_final - r_init) / value[-1]))) + 1

            if not self._allow_rounding_of_stepsize:
                dr = np.sign(r_final - r_init) * abs(value[-1]) * (num - 1)
                r_final = r_init + dr

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
