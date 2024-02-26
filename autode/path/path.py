import numpy as np

from autode.species import Species
from autode.input_output import atoms_to_xyz_file
from autode.log import logger
from autode.units import KcalMol

from typing import Optional


class Path(list):
    def __init__(self, *args: Species, units=KcalMol):
        """
        Base path class that may be populated with species or nudged elastic
        band images, *must* have .energy attributes

        -----------------------------------------------------------------------
        Arguments:
            args (autode.species.species.Species):

        Keyword Arguments:
            units (autode.units.Unit):
        """
        super().__init__()

        for arg in args:
            assert isinstance(arg, Species)
            self.append(arg)

        self.units = units

    def __eq__(self, other):
        """Are two paths equal?"""
        if not isinstance(other, Path):
            return False

        return list.__eq__(self, other)

    @property
    def energies(self) -> np.ndarray:
        """
        Numpy array of energy for each species/image in this path

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray):
        """
        return np.array([item.energy for item in self])

    @property
    def rel_energies(self) -> np.ndarray:
        """
        "Relative energies in a particular unit

        Returns:
            (np.ndarray):
        """
        if len(self) == 0:
            logger.warning("Cannot determine relative energies with no points")
            return np.array([])

        return self.units.times * (self.energies - np.min(self.energies))

    @property
    def peak_idx(self) -> Optional[int]:
        """Get the index of the highest energy peak in this path

        Returns:
             (int | None)
        """
        if any(item.energy is None for item in self):
            logger.warning("An energy was None - cannot locate peak")
            return None

        peaks = [i for i in range(1, len(self) - 1) if self.is_saddle(i)]

        if len(peaks) > 0:
            peak_rel_es = self.rel_energies[np.array(peaks, dtype=int)]
            logger.info(
                f"Found peaks at {peaks} with relative energies "
                f"∆E = {np.round(peak_rel_es, 1)} kcal mol-1"
            )

        # Return the highest energy peak i.e. sorted high -> low
        for peak_idx in sorted(peaks, key=lambda i: -self.energies[i]):
            return peak_idx

        return None

    @property
    def contains_peak(self) -> bool:
        return self.peak_idx is not None

    def product_idx(self, product: "Species") -> Optional[int]:
        """
        Get the index of the point in the path at which products are made.
        If they are not made or they cannot be checked then return None

        -----------------------------------------------------------------------
        Arguments:
            product (autode.species.Species):

        Returns:
            (int | None):
        """
        if product is None or product.graph is None:
            logger.warning("Cannot check if products are made")
            return None

        for i, point in enumerate(self):
            if product.graph.is_isomorphic_to(point.graph):
                logger.info(f"Products made at point {i}")
                return i

        return None

    def products_made(self, product: "Species") -> bool:
        """Are the products are made on the surface?

        -----------------------------------------------------------------------
        Arguments:
            product (autode.species.Species):

        Returns:
            (bool):
        """
        return self.product_idx(product) is not None

    def is_saddle(self, idx: int) -> bool:
        """Is an index a saddle point"""
        if idx == 0 or idx == len(self) - 1:
            logger.warning("Cannot be saddle point, index was at the end")
            return False

        if any(self[i].energy is None for i in (idx - 1, idx, idx + 1)):
            logger.error(
                f"Could not determine if point {idx} was a saddle "
                f"point, an energy close by was None"
            )
            return False

        energy = self[idx].energy
        return self[idx - 1].energy < energy and self[idx + 1].energy < energy

    def plot_energies(
        self, save: bool, name: str, color: str, xlabel: str
    ) -> None:
        """Plot this path"""
        import matplotlib.pyplot as plt

        if len(self) == 0 or any(item.energy is None for item in self):
            logger.error("Could not plot a surface, an energy was None")
            return

        # Plot the relative energies each iteration as a color gradient
        rel_es = self.rel_energies
        plt.plot(np.arange(len(self)), rel_es, marker="o", color=color)

        plt.ylim(-0.1 * np.max(rel_es), 1.1 * np.max(rel_es))
        plt.xlabel(xlabel)
        plt.ylabel(f"∆$E$ / {self.units.name}")
        plt.tight_layout()

        if save:
            plt.savefig(f"{name}.pdf")
            plt.close()

        return None

    def print_geometries(self, name: str) -> None:
        """Print an xyz trajectory of the geometries in the path"""

        open(f"{name}.xyz", "w").close()  # Empty the file

        for i, image in enumerate(self):
            energy = image.energy if image.energy is not None else "none"

            title_line = (
                f"autodE path point {i}. E = {energy} "
                f"charge = {image.charge} "
                f"mult = {image.mult} "
            )

            if image.solvent is not None:
                title_line += f"solvent = {image.solvent.name} "

            atoms_to_xyz_file(
                image.atoms,
                f"{name}.xyz",
                title_line=title_line,
                append=True,
            )
        return None
