from abc import ABC, abstractmethod
from time import sleep
from autode.log import logger
from autode.exceptions import NoClosestSpecies
from copy import deepcopy
import numpy as np
from itertools import product


def get_closest_species(pes, indexes):
    """
    Given a point on an n-dimensional potential energy surface defined by indices where the length is the dimension
    of the surface

    Arguments:
        pes (autode.pes.PES): Potential energy surface
        indexes (tuple): Index of the current point

    Returns:
        (tuple): Index(es) of the closest
    """

    if all(index == 0 for index in indexes):
        logger.info('PES is at the first point')
        return deepcopy(pes.species[indexes])

    # The indcies of the nearest and second nearest points to e.g. n,m in a 2 dimensional PES
    neareast_neighbours = [-1, 0, 1]
    next_nearest_neighbours = [-2, -1, 0, 1, 2]

    # First attempt to find a species that has been calculated in the nearest neighbours
    for index_array in [neareast_neighbours, next_nearest_neighbours]:

        # Each index array has elements from the most negative to most positive. e.g. (-1, -1), (-1, 0) ... (1, 1)
        for d_indexes in product(index_array, repeat=len(indexes)):

            # For e.g. a 2D PES the new index is (n+i, m+j) where i, j = d_indexes
            new_indexes = tuple(np.array(indexes) + np.array(d_indexes))

            try:
                if pes.species[new_indexes] is not None:
                    logger.info(f'Closest point in the PES has indices {new_indexes}')
                    return deepcopy(pes.species[new_indexes])

            except IndexError:
                logger.warning('Closest point on the PES was outside the PES')

    logger.error(f'Could not get a close point to {indexes}')
    raise NoClosestSpecies


class PES(ABC):

    @abstractmethod
    def get_atoms_saddle_point(self):
        pass

    @abstractmethod
    def get_energy_saddle_point(self):
        pass

    @abstractmethod
    def calculate(self, name, method, keywords):
        pass

    @abstractmethod
    def _get_saddle_point_index(self):
        pass

    species = None
    rs = None
