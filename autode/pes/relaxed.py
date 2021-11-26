import numpy as np
import itertools as it
from typing import Tuple, List
from autode.log import logger
from autode.config import Config
from autode.utils import NoDaemonPool
from autode.pes.pes_nd import PESnD
from autode.calculation import Calculation
from autode.exceptions import CalculationException


class RelaxedPESnD(PESnD):

    def _calculate(self) -> None:
        """
        Calculate the n-dimensional surface
        """

        for points in self._points_generator():
            logger.info(f'Calculating tranche {points} on the surface')

            with NoDaemonPool(processes=Config.n_cores) as pool:

                results = []
                for point in points:
                    m = self._species.new_species(name=self._point_name(point))
                    m.coordinates = self._closest_coordinates(point)
                    m.constraints.distance = self._constraints(point)

                    results.append(pool.apply_async(func=_energy_coordinates,
                                                    args=(self, m))
                                   )

                for i, point in enumerate(points):
                    print(point)
                    (self._energies[point],
                     self._coordinates[point]) = results[i].get(timeout=None)

        return None

    def _single_energy_coordinates(self, species) -> Tuple[float, np.ndarray]:
        """Calculate a single energy and set of coordinates on this surface"""

        const_opt = Calculation(name=species.name,
                                molecule=species,
                                method=self._method,
                                n_cores=self._n_cores,
                                keywords=self._keywords)

        try:
            species.optimise(method=self._method, calc=const_opt)
            return float(species.energy), np.array(species.coordinates)

        except CalculationException:
            logger.error(f'Optimisation failed for: {species.name}')
            return np.nan, np.zeros(shape=(species.n_atoms, 3))

    def _default_keywords(self,
                          method: 'autode.wrapper.ElectronicStructureMethod'
                          ) -> 'autode.wrappers.Keywords':
        """Default keywords"""
        return method.keywords.opt

    def _closest_coordinates(self,
                             point: Tuple) -> np.ndarray:
        """
        From a point in the PES defined by its indices obtain the closest set
        of coordinates, which to use as a starting guess for the constrained
        optimisation of this point. The closest point is obtained by computing
        all distances to the n^th nearest neighbours that also has an energy.

        -----------------------------------------------------------------------
        Arguments:
            point: Tuple of indicies in the surface e.g. (0,) in a 1D surface
                   or (0, 1, 2) in a 3D surface

        Returns:
            (np.ndarray): Coordinates. shape = (n_atoms, 3)
        """
        if point == self.origin:
            return self._coordinates[self.origin]

        # Increment out from the nearest neighbours ('distance' 1)
        for n in range(1, max(self.shape)):

            # Construct a ∆-point tuple, which can be added to the current
            # point to generate one close by, which may have an energy and thus
            # should be selected
            for d_point in it.product(range(-n, n+1), repeat=self.ndim):

                close_point = tuple(np.array(point) + np.array(d_point))

                if not self._point_is_contained(close_point):
                    continue

                if self._point_has_energy(close_point):
                    return self._coordinates[close_point]

        raise RuntimeError('Failed to find coordinates with an associated '
                           f'energy close to point {point} in the PES')

    def _constraints(self,
                     point: Tuple) -> dict:
        """
        Construct the distance constraints required for a particular point
        on the PES

        -----------------------------------------------------------------------
        Arguments:
            point: Indicied of a point on the surface

        Returns:
            (dict): Distance constraints
        """
        if not self._point_is_contained(point):
            raise ValueError(f'Cannot determine constraints for a point: '
                             f'{point} in a {self.ndim}D-PES')

        return {r.atom_idxs: r[idx] for r, idx in zip(self._rs, point)}

    def _points_generator(self) -> List[Tuple]:
        """
        Yield points on this surface that sum to the same total, thus are
        close and should be calculated in a group, in parallel. This *should*
        provide the most efficient calculation decomposition on the surface

        -----------------------------------------------------------------------
        Yields:
            (list(tuple(int))):
        """
        all_points = list(self._points())

        for i in range(0, sum(self.shape)):

            points = []
            while all_points:

                # Next point is the next step in the grid
                if len(points) > 0 and sum(all_points[0]) > i:
                    break

                points.insert(0, all_points.pop(0))

            yield points

        return StopIteration


def _energy_coordinates(pes, species):
    return pes._single_energy_coordinates(species)
