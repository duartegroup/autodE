from copy import deepcopy
import numpy as np
from autode.transition_states.ts_guess import get_ts_guess
from autode.config import Config
from autode.log import logger
from autode.mol_graphs import is_isomorphic
from autode.mol_graphs import make_graph
from autode.plotting import plot_1dpes
from autode.pes.pes import get_closest_species
from autode.pes.pes import get_point_species
from autode.pes.pes import PES
from autode.units import KcalMol
from autode.utils import work_in


class PES1d(PES):

    def get_species_saddle_point(self):
        """Get the possible first order saddle points, which are just the
        peaks in the PES"""
        energies = [self.species[i].energy for i in range(self.n_points)]

        # Peaks have lower energies both sides of them
        peaks = [i for i in range(1, self.n_points - 1) if energies[i-1] < energies[i] and energies[i+1] < energies[i]]

        # Yield the peak with the highest energy first
        for peak in sorted(peaks, key=lambda p: -self.species[p].energy):
            yield self.species[peak]

        return None

    def print_plot(self, method_name, name='PES1d'):
        """Print a 1D surface using matplotlib"""
        min_energy = min([species.energy for species in self.species])
        rel_energies = [KcalMol.conversion * (species.energy - min_energy) for species in self.species]

        return plot_1dpes(self.rs, rel_energies, name=name, method_name=method_name)

    def products_made(self):
        logger.info('Checking that somewhere on the surface product(s) are made')

        for i in range(self.n_points):
            make_graph(self.species[i])

            if is_isomorphic(graph1=self.species[i].graph, graph2=self.product_graph):
                logger.info(f'Products made at point {i} in the 1D surface')
                return True

        return False

    @work_in('pes1d')
    def calculate(self, name, method, keywords):
        """Calculate all the points on the surface in serial using the maximum
         number of cores available"""

        for i in range(self.n_points):
            closest_species = get_closest_species((i,), self)

            # Set up the dictionary of distance constraints keyed with bond
            # indexes and values the current r1, r2.. value
            distance_constraints = {self.rs_idxs[0]: self.rs[i][0]}

            self.species[i] = get_point_species((i,), closest_species, distance_constraints, name, method, keywords, Config.n_cores)

        return None

    def __init__(self, reactant, product, rs, r_idxs):
        """
        A one dimensional potential energy surface

        Arguments:
            reactant (autode.complex.ReactantComplex): Species at rs[0]
            product (autode.complex.ProductComplex):
            rs (np.ndarray): Bond length array
            r_idxs (tuple): Atom indexes that the PES will be calculated over
        """
        self.n_points = len(rs)
        self.rs = np.array([(r, ) for r in rs])

        # Vector to store the species
        self.species = np.empty(shape=(self.n_points,), dtype=object)
        self.species[0] = deepcopy(reactant)

        # Tuple of the atom indices scanned in coordinate r
        self.rs_idxs = [r_idxs]

        # Molecular graph of the product. Used to check that the products have
        # been made & find the MEP
        self.product_graph = product.graph


def get_ts_guess_1d(reactant, product, bond, name, method, keywords, dr=0.1):
    """Scan the distance between two atoms and return a guess for the TS

    Arguments:
        reactant (autode.complex.ReactantComplex):
        product (autode.complex.ProductComplex):
        bond (autode.pes.ScannedBond):
        name (str): name of reaction
        method (autode.): electronic structure wrapper to use for the calcs
        keywords (autode.keywords.Keywords): keywords to use in the calcs

    Keyword Arguments:
        dr (float): Δr on the surface *absolute value* in angstroms

    Returns:
        (autode.transition_states.ts_guess.TSguess)
    """
    logger.info(f'Getting TS guess from 1D relaxed potential energy scan using '
                f'{bond.atom_indexes} as the active bond')

    # Create a potential energy surface in the active bonds and calculate
    pes = PES1d(reactant=reactant, product=product,
                rs=np.arange(bond.curr_dist, bond.final_dist, step=dr if bond.final_dist > bond.curr_dist else -dr),
                r_idxs=bond.atom_indexes)

    pes.calculate(name=name, method=method, keywords=keywords)

    if not pes.products_made():
        logger.error('Products were not made on the whole PES')
        return None

    # Plot the line using matplotlib
    pes.print_plot(name=name, method_name=method.name)

    for species in pes.get_species_saddle_point():
        return get_ts_guess(species=species, reactant=reactant, product=product, name=name)

    logger.error('No possible TSs found on the 1D surface')
    return None
