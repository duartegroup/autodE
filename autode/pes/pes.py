from abc import ABC
from abc import abstractmethod
from copy import deepcopy
import itertools
import numpy as np
from autode.calculation import Calculation
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoClosestSpecies
from autode.log import logger


def get_closest_species(point, pes):
    """
    Given a point on an n-dimensional potential energy surface defined by
    indices where the length is the dimension of the surface

    Arguments:
        pes (autode.pes.PES): Potential energy surface
        point (tuple): Index of the current point

    Returns:
        (autode.species.Species): Species
    """

    if all(index == 0 for index in point):
        logger.info('PES is at the first point')
        return deepcopy(pes.species[point])

    # The indcies of the nearest and second nearest points to e.g. n,m in a 2
    # dimensional PES
    neareast_neighbours = [-1, 0, 1]
    next_nearest_neighbours = [-2, -1, 0, 1, 2]

    # First attempt to find a species that has been calculated in the nearest
    # neighbours
    for index_array in [neareast_neighbours, next_nearest_neighbours]:

        # Each index array has elements from the most negative to most
        # positive. e.g. (-1, -1), (-1, 0) ... (1, 1)
        for d_indexes in itertools.product(index_array, repeat=len(point)):

            # For e.g. a 2D PES the new index is (n+i, m+j) where
            # i, j = d_indexes
            new_point = tuple(np.array(point) + np.array(d_indexes))

            try:
                if pes.species[new_point] is not None:
                    logger.info(f'Closest point in the PES has indices '
                                f'{new_point}')
                    return deepcopy(pes.species[new_point])

            except IndexError:
                logger.warning('Closest point on the PES was outside the PES')

    logger.error(f'Could not get a close point to {point}')
    raise NoClosestSpecies


def get_point_species(point, species, distance_constraints, name, method,
                      keywords, n_cores, energy_threshold=1):
    """
    On a PES calculate the energy and the structure using a constrained
    optimisation

    Arguments:
        point (tuple(int)): Index of this point e.g. (0, 0) for the first point
                            on a 2D surface

        species (autode.species.Species):

        distance_constraints (dict): Keyed with atom indexes and the constraint
                             value as the value

        name (str):

        method (autode.wrappers.base.ElectronicStructureMethod):

        keywords (autode.wrappers.keywords.Keywords):

        n_cores (int): Number of cores to used for this calculation

    Keyword Arguments:
        energy_threshold (float): Above this energy (Ha) the calculation
                                  will be disregarded

    Returns:
        (autode.species.Species): Species
    """
    logger.info(f'Calculating point {point} on PES surface')

    original_species = species.copy()
    p_species = species.new_species(name=f'{name}_scan_{"-".join([str(p) for p in point])}')

    # Set up and run the calculation
    const_opt = Calculation(name=p_species.name, molecule=p_species, method=method,
                            n_cores=n_cores,
                            keywords=keywords,
                            distance_constraints=distance_constraints)
    try:
        p_species.optimise(method=method, calc=const_opt)

    except AtomsNotFound:
        logger.error(f'Optimisation failed for {point}')
        return original_species

    # If the energy difference is > 1 Hartree then likely something has gone
    # wrong with the EST method we need to be not on the first point to compute
    # an energy difference..
    if not all(p == 0 for p in point):
        if species.energy is None or np.abs(species.energy - p_species.energy) > energy_threshold:
            logger.error(f'PES point had a relative energy '
                         f'> {energy_threshold} Ha. Using the closest')
            return original_species

    return p_species


class PES(ABC):

    @abstractmethod
    def get_species_saddle_point(self, *args):
        """Return the autode.species.Species at the saddle point"""
        pass

    @abstractmethod
    def products_made(self):
        """Have the products been made somewhere on the surface?"""
        pass

    @abstractmethod
    def calculate(self, name, method, keywords):
        """Calculate all energies and optimised geometries on the surface"""
        pass

    species = None
    rs = None
    rs_idxs = None
