import numpy as np
from copy import deepcopy
from autode.config import Config
from autode.log import logger
from autode.pes import PES
from autode.pes import get_point_species
from autode.transition_states.ts_guess import TSguess
from autode.plotting import plot_1dpes
from autode.units import KcalMol
from autode import mol_graphs
from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm


class PES1d(PES):

    def get_species_saddle_point(self):
        """Get the possible first order saddle points, which are just the peaks in the PES"""

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

    def products_made(self, product):
        logger.info('Checking that somewhere on the surface product(s) are made')

        for i in range(self.n_points):
            mol_graphs.make_graph(self.species[i])

            if mol_graphs.is_isomorphic(graph1=self.species[i].graph, graph2=product.graph):
                logger.info(f'Products made at point {i} in the 1D surface')
                return True

        return False

    def calculate(self, name, method, keywords):
        """Calculate all the points on the surface in serial using the maximum number of cores available"""

        for i in range(self.n_points):
            self.species[i] = get_point_species((i,), self, name, method, keywords, Config.n_cores)

        return None

    def __init__(self, reactant, rs, r_idxs):
        """
        A one dimensional potential energy surface

        Arguments:
            reactant (autode.complex.ReactantComplex): Species at rs[0]
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


def get_ts_guess_1d(reactant, product, active_bond, n_steps, name, method, keywords, final_dist):
    """Scan the distance between two atoms and return a guess for the TS

    Arguments:
       reactant (autode.complex.ReactantComplex):
        product (autode.complex.ProductComplex):
        active_bond1 (tuple): tuple of atom ids showing the first bond being scanned
        active_bond2 (tuple): tuple of atom ids showing the second bond being scanned
        n_steps (int): number of steps to take for each bond in the scan (so n^2 differenct scan points in total)
        name (str): name of reaction
        method (autode.): electronic structure wrapper to use for the calcs
        keywords (list): keywords_list to use in the calcs
        final_dist (float): distance to add onto the current distance of active_bond1 (Å) in n_steps (default: {1.5})

    Returns:
        (autode.transition_states.ts_guess.TSguess)
    """
    logger.info(f'Getting TS guess from 1D relaxed potential energy scan using {active_bond} as the active bond')
    curr_dist = reactant.get_distance(atom_i=active_bond[0], atom_j=active_bond[1])

    # Create a potential energy surface in the active bonds and calculate
    pes = PES1d(reactant=reactant,
                rs=np.linspace(curr_dist, final_dist, n_steps), r_idxs=active_bond)

    pes.calculate(name=name, method=method, keywords=keywords)

    if not pes.products_made(product):
        logger.error('Products were not made on the whole PES')
        return None

    # Fit an analytic 2D PES to the surface and plot using matplotlib
    pes.print_plot(name=name, method_name=method.name)

    # Yield a TSGuess for every saddle point on the surface
    for species in pes.get_species_saddle_point():
        return TSguess(atoms=species.atoms, reactant=reactant, product=product, name=name)

    logger.error('No possible TSs found on the 1D surface')
    return None
