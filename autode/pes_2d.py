from copy import deepcopy
from numpy.polynomial import polynomial
import numpy as np
import multiprocessing.pool
import multiprocessing
from autode.transition_states.ts_guess import get_ts_guess
from autode.calculation import Calculation
from autode.config import Config
from autode.exceptions import AtomsNotFound
from autode.log import logger
from autode.methods import high_level_method_names
from autode.min_energy_pathway import get_sum_energy_mep
from autode.mol_graphs import is_isomorphic
from autode.mol_graphs import make_graph
from autode.pes import get_point_species
from autode.pes import get_closest_species
from autode.pes import PES
from autode.plotting import plot_2dpes
from autode.saddle_points import poly2d_saddlepoints
from autode.utils import work_in
from autode.units import KcalMol


class PES2d(PES):

    def get_species_saddle_point(self, name, method, keywords):
        """Get the species at the true saddle point on the surface"""
        saddle_points = poly2d_saddlepoints(coeff_mat=self.coeff_mat, xs=self.r1s, ys=self.r2s)

        logger.info('Sorting the saddle points by their minimum energy path to reactants and products')
        saddle_points = sorted(saddle_points, key=lambda s: get_sum_energy_mep(s, self))

        for saddle_point in saddle_points:
            r1, r2 = saddle_point

            # Determine the indicies of the point closest to the analytic saddle point to use as a guess
            close_point = (np.argmin(np.abs(self.r1s - r1)), np.argmin(np.abs(self.r2s - r2)))
            logger.info(f'Closest point is {close_point} with r1 = {self.rs[close_point][0]:.3f}, '
                        f'r2 = {self.rs[close_point][1]:.3f} Å')

            # Perform a constrained optimisation using the analytic saddle point r1, r2 values
            species = deepcopy(self.species[close_point])
            const_opt = Calculation(name=f'{name}_const_opt', molecule=species, method=method,
                                    opt=True, n_cores=Config.n_cores, keywords_list=keywords,
                                    distance_constraints={self.rs_idxs[0]: r1, self.rs_idxs[1]: r2})

            try:
                species.run_const_opt(const_opt, method, Config.n_cores)
            except AtomsNotFound:
                logger.error('Constrained optimisation at the saddle point failed')
                pass

            return species

        return None

    def fit(self, polynomial_order):
        """Fit an analytic 2d surface"""

        # Compute a flat list of relative energies to use to fit the polynomial
        min_energy = min([species.energy for species in self.species.flatten()])
        rel_energies = [KcalMol.conversion * (species.energy - min_energy) for species in self.species.flatten()]

        # Compute the polynomial_order x polynomial_order matrix of coefficients
        self.coeff_mat = polyfit2d(x=[r[0] for r in self.rs.flatten()],
                                   y=[r[1] for r in self.rs.flatten()],
                                   z=rel_energies, order=polynomial_order)
        return None

    def print_plot(self, name='PES2d'):
        """Print the fitted 2D surface using matplotlib"""
        return plot_2dpes(coeff_mat=self.coeff_mat, r1=self.r1s, r2=self.r2s, name=name)

    def products_made(self):
        """Check that somewhere on the surface the molecular graph is isomorphic to the product"""
        logger.info('Checking that somewhere on the surface product(s) are made')

        for i in range(self.n_points_r1):
            for j in range(self.n_points_r2):
                make_graph(self.species[i, j])

                if is_isomorphic(graph1=self.species[i, j].graph, graph2=self.product_graph):
                    logger.info(f'Products made at ({i}, {j})')
                    return True

        return False

    @work_in('pes2d')
    def calculate(self, name, method, keywords):
        """Calculations on the surface with a method using the a decomposition similar to the following

        Calculation order            Indexes

           4  5  6  7          .        .        .

           3  4  5  6        (0, 2)     .        .

           2  3  4  5        (0, 1)  (1, 1)      .

           1  2  3  4        (0, 0)  (1, 0)    (2, 0)
                                ↖       ↖         ↖
                            sum = 0   sum = 1    sum = 2

        Arguments:
            name (str):
            method (autode.wrappers.ElectronicStructureMethod):
            keywords (list(str)):
        """
        logger.info(f'Running a 2D PES scan with {method.name}. {self.n_points_r1*self.n_points_r2} total points')

        for sum_indexes in range(self.n_points_r1 + self.n_points_r2 - 1):

            all_points = [(i, j) for i in range(self.n_points_r1) for j in range(self.n_points_r2)]

            # Strip those that are along the current diagonal – the sum of indices is constant
            diagonal_points = [point for point in all_points if sum(point) == sum_indexes]

            # Strip the indices that are not in the array. This applies if n_points_r1 != n_points_r2
            points = [(i, j) for (i, j) in diagonal_points if i < self.n_points_r1 and j < self.n_points_r2]

            # The cores for this diagonal are the floored number of total cores divided by the number of calculations
            cores_per_process = Config.n_cores // len(points) if Config.n_cores // len(points) > 1 else 1

            closest_species = [get_closest_species(p, self) for p in points]
            dimension = len(self.rs_idxs)
            # Set up the dictionary of distance constraints keyed with bond indexes and values the current r1, r2. value
            distance_constraints = [{self.rs_idxs[i]: self.rs[p][i] for i in range(dimension)} for p in points]

            # Use custom NoDaemonPool here, as there are several multiprocessing events happening within the function
            with NoDaemonPool(processes=Config.n_cores) as pool:
                results = [pool.apply_async(func=get_point_species, args=(p, s, d, name, method, keywords, cores_per_process))
                           for p, s, d in zip(points, closest_species, distance_constraints)]

                for i, point in enumerate(points):
                    self.species[point] = results[i].get(timeout=None)

        logger.info('2D PES scan done')
        return None

    def _init_tensors(self, reactant, r1s, r2s):
        """Initialise the matrices of Species and distances"""
        logger.info(f'Initialising the {len(r1s)}x{len(r2s)} PES matrices')

        assert self.rs.shape == self.species.shape

        for i in range(len(self.rs)):
            for j in range(len(self.rs[i])):
                # Tuple of distances
                self.rs[i, j] = (r1s[i], r2s[j])

        # Copy of the reactant complex, whose atoms/energy will be set in the scan
        self.species[0, 0] = deepcopy(reactant)

        return None

    def __init__(self, reactant, product, r1s, r1_idxs, r2s, r2_idxs):
        """
        A two dimensional potential energy surface

           |
        r2 |
           |
           |___________
                r1

        Arguments:
            reactant (autode.complex.ReactantComplex): Species at r1s[0] and r2s[0]
            r1s (np.ndarray): Bond length array in r1
            r1_idxs (tuple): Atom indexes that the PES will be calculated over in r1
            r2s (np.ndarray): Bond length array in r2
            r2_idxs (tuple): Atom indexes that the PES will be calculated over in r2
        """
        self.r1s, self.r2s = r1s, r2s
        self.n_points_r1, self.n_points_r2 = len(r1s), len(r2s)

        # Matrices to store the species and r1, r2 values at a point (i, j)
        self.species = np.empty(shape=(self.n_points_r1, self.n_points_r2), dtype=object)
        self.rs = np.empty(shape=(self.n_points_r1, self.n_points_r2), dtype=tuple)

        # List of tuples that contain atom indices of the coordinate r1
        self.rs_idxs = [r1_idxs, r2_idxs]

        # Populate the rs array and set species[0, 0] to the reactant
        self._init_tensors(reactant=reactant, r1s=r1s, r2s=r2s)

        # Coefficients of the fitted surface
        self.coeff_mat = None

        # Molecular graph of the product. Used to check that the products have been made & find the MEP
        self.product_graph = product.graph


def get_ts_guess_2d(reactant, product, bond1, bond2, name, method, keywords, polynomial_order=3, dr=0.1):
    """Scan the distance between two sets of two atoms and return a guess for the TS

    Arguments:
        reactant (autode.complex.ReactantComplex):
        product (autode.complex.ProductComplex):
        bond1 (autode.pes.ScannedBond):
        bond2 (autode.pes.ScannedBond):
        name (str): name of reaction
        method (autode.): electronic structure wrapper to use for the calcs
        keywords (list): keywords_list to use in the calcs

    Keyword Arguments:
        polynomial_order (int): order of polynomial to fit the data to (default: {3})
        dr (float): Δr on the surface *absolute value*

    Returns:
        (autode.transition_states.ts_guess.TSguess)
    """
    logger.info(f'Getting TS guess from 2D relaxed potential energy scan, using active bonds '
                f'{bond1} and {bond2}')

    # Steps of +Δr if the final distance is greater than the current else -Δr. Run at least a 3x3 PES
    n_steps1 = max(int(np.abs((bond1.final_dist - bond1.curr_dist) / dr)), 3)
    n_steps2 = max(int(np.abs((bond2.final_dist - bond2.curr_dist) / dr)), 3)

    if method.name in high_level_method_names:
        logger.warning('Limiting the number of steps to a maximum of 8 so <64 high level optimisations have to be done')
        n_steps1 = min(n_steps1, 8)
        n_steps2 = min(n_steps2, 8)

    # Create a potential energy surface in the two active bonds and calculate
    pes = PES2d(reactant=reactant, product=product,
                r1s=np.linspace(bond1.curr_dist, bond1.final_dist, n_steps1), r1_idxs=bond1.atom_indexes,
                r2s=np.linspace(bond2.curr_dist, bond2.final_dist, n_steps2), r2_idxs=bond2.atom_indexes)

    pes.calculate(name=name, method=method, keywords=keywords)

    if not pes.products_made():
        logger.error('Products were not made on the whole PES')
        return None

    # Fit an analytic 2D PES to the surface and plot using matplotlib
    pes.fit(polynomial_order=polynomial_order)
    pes.print_plot(name=name)

    # Get a TSGuess for the lowest energy MEP saddle point on the surface
    species = pes.get_species_saddle_point(name=name, method=method, keywords=keywords)

    if species is not None:
        return get_ts_guess(species=species, reactant=reactant, product=product, name=name)

    logger.error('No possible TSs found on the 2D surface')
    return None


def polyfit2d(x, y, z, order):
    """Takes x and y coordinates and their resultant z value, and creates a matrix where element i,j is the coefficient
    of the desired order polynomial x ** i * y ** j

    Arguments:
        x (np.array): flat array of x coordinates
        y (np.array): flat array of y coordinates
        z (np.array): flat array of z value at the corresponding x and y value
        order (int): max order of polynomial to work out

    Returns:
        np.array: matrix of polynomial coefficients
    """
    logger.info('Fitting 2D surface to polynomial in x and y')
    deg = np.array([int(order), int(order)])
    vander = polynomial.polyvander2d(x, y, deg)
    # vander matrix is matrix where each row i deals with x=x[i] and y=y[i], and each item in the
    # row has value x ** m * y ** n with (m,n) = (0,0), (0,1), (0,2) ... (1,0), (1,1), (1,2) etc up to (order, order)
    coeff_mat, _, _, _ = np.linalg.lstsq(vander, z, rcond=None)
    return coeff_mat.reshape(deg + 1)


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)
