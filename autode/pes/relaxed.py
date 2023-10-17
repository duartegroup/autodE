import numpy as np
import itertools as it

from typing import Tuple, List, Type, Iterator, TYPE_CHECKING

from autode.log import logger
from autode.utils import hashable, ProcessPool
from autode.pes.reactive import ReactivePESnD
from autode.constraints import DistanceConstraints
from autode.calculations import Calculation
from autode.exceptions import CalculationException

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.keywords import Keywords
    from autode.wrappers.methods import Method


class RelaxedPESnD(ReactivePESnD):
    """Potential energy surface over a set of distances, where all other
    degrees of freedom are minimised"""

    def _calculate(self) -> None:
        """
        Calculate the n-dimensional surface
        """
        assert self._coordinates is not None, "Coordinates must be set"

        for points in self._points_generator():
            n_cores_pp = max(self._n_cores // len(points), 1)
            logger.info(
                f"Calculating tranche {points} on the surface, using "
                f"{n_cores_pp} cores per process"
            )

            with ProcessPool(max_workers=self._n_cores) as pool:
                func = hashable("_single_energy_coordinates", self)

                jobs = [
                    pool.submit(
                        func, self._species_at(point), n_cores=n_cores_pp
                    )
                    for point in points
                ]

                for i, point in enumerate(points):
                    (
                        self._energies[point],
                        self._coordinates[point],
                    ) = jobs[i].result()

        return None

    @property
    def _default_keyword_type(self) -> Type["Keywords"]:
        from autode.wrappers.keywords import OptKeywords

        return OptKeywords

    def _species_at(self, point: Tuple) -> "Species":
        """
        Generate a species on the PES at a defined point. Attributes are
        obtained from the internal species (molecule at the origin in the PES)
        while the coordinates are set from the closest point and the
        constraints defined by the point.

        -----------------------------------------------------------------------
        Arguments:
            point: Point at which to generate the species e.g. (0,) in a 1D
                   surface or (1, 2 3) for a 3D surface

        Returns:
            (autode.species.Species): Species
        """
        assert self._species

        species = self._species.new_species(name=self._point_name(point))
        species.coordinates = self._closest_coordinates(point)
        species.constraints.distance = self._constraints(point)

        return species

    def _single_energy_coordinates(
        self, species: "Species", **kwargs
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate a single energy and set of coordinates on this surface

        -----------------------------------------------------------------------
        Arguments:
            species: Species on which to perform a constrained minimisation

        Keyword Arguments:
            n_cores: Number of cores to use for the calculation, if left
                     unassigned then use self._n_cores
        """
        assert self._keywords is not None and self._method is not None

        const_opt = Calculation(
            name=species.name,
            molecule=species,
            method=self._method,
            n_cores=kwargs.get("n_cores", self._n_cores),
            keywords=self._keywords,
        )

        try:
            species.optimise(method=self._method, calc=const_opt)
            assert species.energy is not None
            return float(species.energy), np.array(species.coordinates)

        except (CalculationException, ValueError, TypeError, AssertionError):
            logger.error(f"Optimisation failed for: {species.name}")
            return np.nan, np.zeros(shape=(species.n_atoms, 3))

    def _default_keywords(self, method: "Method") -> "Keywords":
        """Default keywords"""
        assert (
            method.keywords.opt is not None
        ), "Method must have optimisation kwds"
        return method.keywords.opt

    def _closest_coordinates(self, point: Tuple) -> np.ndarray:
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
        assert self._coordinates is not None, "Must have set coordinates"

        if point == self.origin:
            return self._coordinates[self.origin]

        # Increment out from the nearest neighbours ('distance' 1)
        for n in range(1, max(self.shape)):
            # Construct a ∆-point tuple, which can be added to the current
            # point to generate one close by, which may have an energy and thus
            # should be selected
            for d_point in it.product(range(-n, n + 1), repeat=self.ndim):
                close_point = tuple(np.array(point) + np.array(d_point))

                if not self._is_contained(close_point):
                    continue

                if self._has_energy(close_point):
                    return self._coordinates[close_point]

        raise RuntimeError(
            "Failed to find coordinates with an associated "
            f"energy close to point {point} in the PES"
        )

    def _constraints(self, point: Tuple) -> DistanceConstraints:
        """
        Construct the distance constraints required for a particular point
        on the PES

        -----------------------------------------------------------------------
        Arguments:
            point: Indices of a point on the surface

        Returns:
            (dict): Distance constraints
        """
        if not self._is_contained(point):
            raise ValueError(
                f"Cannot determine constraints for a point: "
                f"{point} in a {self.ndim}D-PES"
            )

        return DistanceConstraints(
            {r.atom_idxs: r[idx] for r, idx in zip(self._rs, point)}
        )

    def _points_generator(self) -> Iterator[List[Tuple]]:
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
            points: List[tuple] = []
            while all_points:
                # Next point is the next step in the grid
                if len(points) > 0 and sum(all_points[0]) > i:
                    break

                points.insert(0, all_points.pop(0))

            if len(points) > 0:
                yield points

        return StopIteration
