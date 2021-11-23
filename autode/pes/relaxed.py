import numpy as np
from typing import Optional
from autode.log import logger
from autode.config import Config
from autode.pes.pes_nd import PESnD
from autode.calculation import Calculation
from autode.exceptions import CalculationException


class RelaxedPESnD(PESnD):

    def calculate(self,
                  method:   'autode.wrapper.ElectronicStructureMethod',
                  keywords:  Optional['autode.wrappers.Keywords'] = None,
                  n_cores:   Optional[int] = None) -> None:
        """
        Calculate the n-dimensional surface
        """

        if self._species is None:
            raise ValueError('Cannot calculate a PES without an initial '
                             'species. Initialise PESNd with a species '
                             'or reactant')

        for point in self._points():
            logger.info(f'Calculating point {point} on PES surface')

            species = self._species.new_species(name=self._point_name(point))
            species.coordinates = self._closest_coordinates(point)

            const_opt = Calculation(
                        name=species.name,
                        molecule=species,
                        method=method,
                        n_cores=Config.n_cores if n_cores is None else n_cores,
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



