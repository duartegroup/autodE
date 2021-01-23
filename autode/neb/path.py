import numpy as np
import matplotlib.pyplot as plt
from autode import mol_graphs
from autode.input_output import atoms_to_xyz_file
from autode.log import logger
from autode.units import KcalMol


class Path(list):

    @property
    def energies(self):
        """Numpy array of energy for each species/image in this path"""
        return np.array([item.energy for item in self])

    @property
    def rel_energies(self):
        return self.units.conversion * (self.energies - np.min(self.energies))

    @property
    def peak_idx(self):
        """Get the index of the highest energy peak in this path

        Returns:
             (int | None)
        """
        if any(energy is None for energy in self):
            logger.warning('An energy was None - cannot locate peak')
            return None

        peaks = [i for i in range(1, len(self) - 1) if self.is_saddle(i)]

        if len(peaks) > 0:
            peak_rel_es = self.rel_energies[np.array(peaks, dtype=int)]
            logger.info(f'Found peaks at {peaks} with relative energies '
                        f'∆E = {np.round(peak_rel_es, 1)} kcal mol-1')

        # Return the highest energy peak i.e. sorted high -> low
        for peak_idx in sorted(peaks, key=lambda i: -self.energies[i]):
            return peak_idx

        return None

    def products_made(self, product):
        """Check whether the products are made on the surface

        Arguments:
            product (autode.species.Species):

        Returns:
            (bool):
        """
        if product is None or product.graph is None:
            logger.warning('Cannot check if products are made')
            return False

        for i, point in enumerate(self):
            if point.species.graph is None:
                mol_graphs.make_graph(point.species)

            if mol_graphs.is_isomorphic(graph1=point.species.graph,
                                        graph2=product.graph):
                logger.info(f'Products made at point {i}')
                return True

        return False

    def is_saddle(self, idx):
        """Is an index a saddle point"""
        energy = self[idx].energy
        return self[idx-1].energy < energy and self[idx+1].energy < energy

    def plot_energies(self, save, name, color, xlabel):
        """Plot this path"""
        if any(energy is None for energy in self.energies):
            logger.error('Could not plot a surface, an energy was None')
            return

        # Plot the relative energies each iteration as a color gradient
        rel_es = self.rel_energies
        plt.plot(np.arange(len(self)), rel_es, marker='o', color=color)

        plt.ylim(-0.1*np.max(rel_es), 1.1*np.max(rel_es))
        plt.xlabel(xlabel)
        plt.ylabel(f'∆$E$ / {self.units.name}')
        plt.tight_layout()

        if save:
            plt.savefig(f'{name}.png', dpi=300)
            plt.close()

        return None

    def print_geometries(self, name):
        """Print an xyz trajectory of the geometries in the path"""

        # Empty the file
        open(f'{name}.xyz', 'w').close()

        for i, image in enumerate(self):
            assert image.species is not None
            energy = image.energy if image.energy is not None else 'none'

            atoms_to_xyz_file(image.species.atoms,
                              f'{name}.xyz',
                              title_line=f'autodE path point {i}. E = {energy}',
                              append=True)
        return None

    def __init__(self, *args, units=KcalMol):
        """
        Base path class that may be populated with species or nudged elastic
        band images, *must* have .energy attributes

        Arguments:
            args (list(autode.species.Species | autode.neb.Image))
        """
        super().__init__()

        for arg in args:
            assert hasattr(arg, 'energy') and hasattr(arg, 'species')
            self.append(arg)

        self.units = units
