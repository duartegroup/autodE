import numpy as np
import itertools as it
from typing import Optional, Tuple
from autode.log import logger
from autode.pes.pes_nd import PESnD
from autode.calculation import Calculation
from autode.exceptions import CalculationException


class RelaxedPESnD(PESnD):

    def _calculate(self,
                   method:   'autode.wrapper.ElectronicStructureMethod',
                   keywords: Optional['autode.wrappers.Keywords'],
                   n_cores:  int) -> None:
        """
        Calculate the n-dimensional surface
        """

        for point in self._points():
            logger.info(f'Calculating point {point} on PES surface')

            species = self._species.new_species(name=self._point_name(point))
            species.coordinates = self._closest_coordinates(point)

            const_opt = Calculation(
                        name=species.name,
                        molecule=species,
                        method=method,
                        n_cores=n_cores,
                        keywords=keywords,
                        distance_constraints=self._constraints(point)
                        )

            try:
                species.optimise(method=method, calc=const_opt)
                self._coordinates[point] = np.array(species.coordinates,
                                                    copy=True)
                self._energies[point] = float(species.energy)

            except CalculationException:
                logger.error(f'Optimisation failed for {point}')
                self._energies[point] = np.nan

        return None

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
