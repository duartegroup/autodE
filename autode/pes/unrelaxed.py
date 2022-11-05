"""Unrelaxed potential energy surfaces"""
import numpy as np
from typing import Tuple, Type
from multiprocessing import Pool
from autode.pes.reactive import ReactivePESnD
from autode.utils import hashable
from autode.log import logger
from autode.mol_graphs import split_mol_across_bond
from autode.exceptions import CalculationException


class UnRelaxedPES1D(ReactivePESnD):
    """1D potential energy surface without minimising other degrees of freedom.
    Only supports over bonds"""

    def _calculate(self) -> None:
        """Calculate this surface, in the maximally parallel way"""
        self._check()
        points = list(self._points())

        # Number of cores per-process depends on the number of points in the
        # PES. The number of workers executing will be at most len(points)
        n_cores_pp = max(self._n_cores // len(points), 1)

        with Pool(processes=self._n_cores) as pool:

            results = []

            for p in points:
                res = pool.apply_async(
                    func=hashable("_single_energy", self),
                    args=(self._species_at(p), n_cores_pp),
                )
                results.append(res)

            for i, p in enumerate(points):
                self._energies[p] = results[i].get(timeout=None)

        return None

    @property
    def _default_keyword_type(self) -> Type["autode.wrappers.Keywords"]:
        from autode.wrappers.keywords import SinglePointKeywords

        return SinglePointKeywords

    def _species_at(self, point: Tuple) -> "autode.species.Species":
        """
        Shift this structure to a point in the surface

        -----------------------------------------------------------------------
        Arguments:
            point: Point on the surface

        Returns:
            (autode.species.species.Species): New species
        """

        species = self._species.new_species(name=self._point_name(point))
        i, j = self._rs[0].atom_idxs

        shift_idxs, _ = split_mol_across_bond(species.graph, bond=(i, j))

        a = i if i in shift_idxs else j
        b = j if i == a else i

        coords = np.array(species.coordinates, copy=True)
        required_r = self._r(point=point, dim=0)

        coords[shift_idxs] += (
            required_r - species.distance(i, j)
        ) * species.atoms.nvector(b, a)

        self._coordinates[point] = coords
        species.coordinates = coords

        return species

    def _check(self) -> None:
        """Check that some attributes have required values"""

        if self.ndim != 1:
            raise NotImplementedError(
                "Cannot calculate an unrelaxed surface "
                "for >1 dimension surfaces"
            )

        atom_idxs = self._rs[0].atom_idxs
        if atom_idxs not in self._species.graph.edges:
            raise ValueError(
                f"Unrelaxed PESs must be over a bond {atom_idxs} "
                f"was not in the list of bonds"
            )

        return None

    def _default_keywords(
        self, method: "autode.wrappers.ElectronicStructureMethod"
    ) -> "autode.wrappers.Keywords":
        """
        Default keywords for an unrelaxed scan that uses single point
        evaluations is

        -----------------------------------------------------------------------
        Arguments:
            method:

        Returns:
            (autode.wrappers.keywords.Keywords):
        """
        return method.keywords.sp

    def _single_energy(
        self, species: "autode.species.Species", n_cores: int
    ) -> float:
        """
        Evaluate the energy using a single point calculation

        -----------------------------------------------------------------------
        Arguments:
            species: Species on the surface

            n_cores: Number of cores to use

        Returns:
            (float): Energy in Ha
        """

        try:
            species.single_point(
                method=self._method, keywords=self._keywords, n_cores=n_cores
            )

            return float(species.energy)

        except (CalculationException, ValueError, TypeError):
            logger.error(f"Single point failed for: {species.name}")
            return np.nan
