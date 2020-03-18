import numpy as np
from numpy.polynomial import polynomial
from copy import deepcopy
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from autode.config import Config
from autode.constants import Constants
from autode.log import logger
from autode.calculation import Calculation
from autode.plotting import plot_2dpes
from autode.plotting import make_reaction_animation
from autode.transition_states.ts_guess import TSguess
from autode.exceptions import AtomsNotFound, NoClosestSpecies
from autode import mol_graphs
from autode.saddle_points import poly2d_saddlepoints
from autode.saddle_points import best_saddlepoint
from autode.min_energy_pathway import get_mep
from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm
from autode.pes import PES
from autode.pes import get_closest_species


def get_point_atoms_energy(point, pes, name, method, keywords, n_cores):
    """On a 2d PES calculate the energy and the structure using a constrained optimisation"""
    logger.info(f'Calculating point {point} on PES surface')
    i, j = point

    species = get_closest_species(pes=pes, indexes=(i, j))

    # Set up and runc the calculation
    const_opt = Calculation(name=f'{name}_scan_{i}-{j}', molecule=species, method=method, opt=True, n_cores=n_cores,
                            keywords_list=keywords, distance_constraints={pes.r1_idxs: pes.rs[i, j][0],
                                                                          pes.r2_idxs: pes.rs[i, j][1]})
    const_opt.run()

    # Attempt to set the atoms and the energy from the calculation, if failed then leave as the closest
    try:
        species.set_atoms(atoms=const_opt.get_final_atoms())
        species.energy = const_opt.get_energy()

    except AtomsNotFound:
        logger.error(f'Optimisation failed for r1={pes.rs[i, j][0]:.3f}, r2={pes.rs[i, j][1]:.3f}')

    return species


class PES2d(PES):

    def _get_saddle_point_index(self):
        return None

    def get_atoms_saddle_point(self):
        return None

    def get_energy_saddle_point(self):
        return None

    def calculate(self, name, method, keywords):
        """Calculations on the surface with a method using the a decomposition similar to the following

        4 5 6 7
        3 4 5 6
        2 3 4 5
        1 2 3 4

        where the first Config.n_cores are performed in serial
        """
        logger.info(f'Running a 2D PES scan with {method.name}. {self.n_points_r1*self.n_points_r2} total points')

        for sum_indexes in range(self.n_points_r1 + self.n_points_r2 - 1):

            # Calculate the list of points on this diagonal
            all_points = [(i, j) for i in range(self.n_points_r1) for j in range(self.n_points_r2)]

            # Strip those that are along the current diagonal – the sum of indices is constant
            diagonal_points = [point for point in all_points if sum(point) == sum_indexes]

            # Strip the indices that are not in the array. This applies if n_points_r1 != n_points_r2
            points = [(i, j) for (i, j) in diagonal_points if i < self.n_points_r1 and j < self.n_points_r2]

            # The cores for this diagonal are the floored number of total cores divided by the number of calculations
            cores_per_process = Config.n_cores // len(points) if Config.n_cores // len(points) > 1 else 1

            with Pool(processes=len(points)) as pool:
                for (i, j) in points:
                    result = pool.apply_async(func=get_point_atoms_energy, args=((i, j), self, name, method,
                                                                                 keywords, cores_per_process))
                    self.species[i, j] = result.get(timeout=None)

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

    def __init__(self, reactant, r1s, r1_idxs, r2s, r2_idxs):
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
        self.n_points_r1, self.n_points_r2 = len(r1s), len(r2s)

        # Matrices to store the species and r1, r2 values at a point i, j
        self.species = np.empty(shape=(self.n_points_r1, self.n_points_r2), dtype=object)
        self.rs = np.empty(shape=(self.n_points_r1, self.n_points_r2), dtype=tuple)

        self.r1_idxs, self.r2_idxs = r1_idxs, r2_idxs
        self._init_tensors(reactant=reactant, r1s=r1s, r2s=r2s)









def get_ts_guess_2d(reactant, product, active_bond1, active_bond2, n_steps, name, method, keywords,
                    final_dist1, final_dist2, e_grid_points=40, polynomial_order=5):
    """Scan the distance between two sets of two atoms and return a guess for the TS

    Arguments:
        reactant (molcule object): reactant complex
        product (molecule object): product complex
        active_bond1 (tuple): tuple of atom ids showing the first bond being scanned
        active_bond2 (tuple): tuple of atom ids showing the second bond being scanned
        n_steps (int): number of steps to take for each bond in the scan (so n^2 differenct scan points in total)
        name (str): name of reaction
        method (object): electronic structure wrapper to use for the calcs
        keywords (list): keywords_list to use in the calcs
        final_dist1 (float): distance to add onto the current distance of active_bond1 (Å) in n_steps (default: {1.5})
        final_dist2 (float): distance to add onto the current distance of active_bond2 (Å) in n_steps (default: {1.5})

    Keyword Arguments:
        e_grid_points (int): number of points along each axis of the energy grid in the saddlepoint finding
                             (default: {40})
        polynomial_order (int): order of polynomial to fit the data to (default: {5})

    Returns:
        ts guess object: ts guess
    """
    logger.info(f'Getting TS guess from 2D relaxed potential energy scan, using active bonds '
                f'{active_bond1} and {active_bond2}')

    curr_dist1 = reactant.get_distance(atom_i=active_bond1[0], atom_j=active_bond1[1])
    curr_dist2 = reactant.get_distance(atom_i=active_bond2[0], atom_j=active_bond2[1])

    # Create a potential energy surface in the two active bonds
    pes = PES2d(reactant=reactant,
                r1s=np.linspace(curr_dist1, final_dist1, n_steps), r1_idxs=active_bond1,
                r2s=np.linspace(curr_dist2, final_dist2, n_steps), r2_idxs=active_bond2)

    pes.calculate(name=name, method=method, keywords=keywords)
    exit()


    dist_grid1, dist_grid2 = np.meshgrid(r1, r2)

    # Create a grid of molecules and associated constrained optimisation calculations
    mol_grid = [[deepcopy(reactant) for _ in range(n_steps)] for _ in range(n_steps)]
    for row in mol_grid:
        for reactant in row:
            reactant.qm_solvent_xyzs = None

    # Perform a 1d scan in serial
    for n in range(n_steps):
        if n == 0:
            molecule = reactant
        else:
            molecule = mol_grid[0][n-1]


        # Set the new xyzs as those output from the calculation, and the previous if no xyzs could be found
        try:
            # Set the new xyzs of the molecule
            mol_grid[0][n].set_atoms(atoms=const_opt.get_final_atoms())
        except AtomsNotFound:
            mol_grid[0][n].set_atoms(atoms=deepcopy(mol_grid[0][n-1].atoms) if n != 0 else deepcopy(reactant.atoms))

            mol_grid[0][n].energy = const_opt.get_energy()

    # Execute the remaining set of optimisations in parallel
    for i in range(1, n_steps):

        if Config.n_cores <= n_steps:
            cores_per_process = 1
        else:
            cores_per_process = Config.n_cores//n_steps
            if Config.n_cores % n_steps != 0:
                logger.warning(f'Not all cores will be used in the multiprocessing stage, for optimal core usage use a '
                               f'multiple of {n_steps} cores')

        calcs = [Calculation(name +f'_scan{i}_{n}', mol_grid[i-1][n], method, n_cores=cores_per_process, opt=True,
                             keywords_list=keywords, distance_constraints={active_bond1: dist_grid1[i][n],
                                                                           active_bond2: dist_grid2[i][n]})
                 for n in range(n_steps)]

        [calc.generate_input() for calc in calcs]
        with Pool(processes=Config.n_cores) as pool:
            results = [pool.apply_async(execute_calc, (calc,)) for calc in calcs]
            [res.get(timeout=None) for res in results]

        [calc.set_output_file_lines() for calc in calcs]

        # Add attributes for molecules in the mol_grid
        for n in range(n_steps):
            calcs[n].terminated_normally = calcs[n].calculation_terminated_normally()
            # Set the new xyzs as those output from the calculation, and the previous if no xyzs could be found
            try:
                mol_grid[i][n].xyzs = calcs[n].get_final_atoms()
            except AtomsNotFound:
                mol_grid[i][n].xyzs = deepcopy(mol_grid[i-1][n].xyzs)


    # Populate the dictionary of distances, xyzs and energies
    # TODO make this nicer
    dist_xyzs_energies = {}
    for n in range(n_steps):
        for m in range(n_steps):
            dist_xyzs_energies[(dist_grid1[n, m], dist_grid2[n, m])] = (mol_grid[n][m].xyzs, mol_grid[n][m].energy)

    # check product and TSGuess product graphs are isomorphic
    expected_prod_graphs = mol_graphs.get_separate_subgraphs(product.graph)
    logger.info('Checking products were made')
    products_made = False
    for row in mol_grid[::-1]:
        ts_product_graphs = [mol_graphs.make_graph(mol.xyzs, len(mol.xyzs)) for mol in row[::-1]]
        for ts_product_graph in ts_product_graphs:
            if all(mol_graphs.is_subgraph_isomorphic(ts_product_graph, graph) for graph in expected_prod_graphs):
                products_made = True
                break
        if products_made:
            break

    if not products_made:
        logger.info('Products not made')
        return None

    # Make a new molecule that will form the basis of the TS guess object
    tsguess_mol = deepcopy(reactant)
    tsguess_mol.name = name
    saddle_point_xyzs_output = find_2dpes_saddlepoint_xyzs(
        dist_xyzs_energies, scan_name=name, plot_name=f'{reactant.name}_{active_bond1[0]}_{active_bond1[1]}_'
                                                      f'{active_bond2[0]}_{active_bond2[1]}_2dscan',
        method=method, n_points=e_grid_points, order=polynomial_order)

    if saddle_point_xyzs_output is None:
        logger.error('No xyzs found for the saddle point')
        return None
    else:
        saddle_point_xyzs, saddlepoint = saddle_point_xyzs_output
    tsguess_mol.set_xyzs(xyzs=saddle_point_xyzs)

    logger.info('Running a constrain optimisation for the analytic saddlepoint distances')
    ts_const_opt = Calculation(name=f'{name}_saddlepoint_opt', molecule=tsguess_mol, method=method,
                               n_cores=Config.n_cores, opt=True, keywords_list=keywords,
                               distance_constraints={active_bond1: saddlepoint[0], active_bond2: saddlepoint[1]})
    ts_const_opt.run()
    try:
        tsguess_mol.set_xyzs(xyzs=ts_const_opt.get_final_atoms())
    except AtomsNotFound:
        pass

    return TSguess(atoms=tsguess_mol.atoms, reactant=reactant, product=product)


def find_2dpes_saddlepoint_xyzs(dists_xyzs_energies_dict, scan_name, plot_name, method, n_points, order):
    """Find the best saddle point on a 2D PES, and the closest xyzs to this point given the grid of distances, xyzs and energies

    Arguments:
        dists_xyzs_energies_dict (dict): [dist] = (xyzs, energy)
        scan_name (str): name of reaction
        plot_name ([type]): name of plot made
        method (object): electronic structure wrapper to use for the calcs

    Keyword Arguments:
        n_points (int): number of points along each axis of the energy grid (default: {40})
        order (int): order of polynomial to fit the data to (default: {5})

    Returns:
        tuple: (saddlepoint xyzs, (r1 saddle coord, r2 saddle coord))
    """

    logger.info('Finding saddle point in 2D PES')

    energies = [dists_xyzs_energies_dict[dists][1]
                for dists in dists_xyzs_energies_dict.keys()]
    # The energy list may have None values in, so to perform the fitting replace with the closest float value
    energies_not_none = list(replace_none(energies))
    flat_rel_energy_array = [Constants.ha2kcalmol *
                             (e - min(energies_not_none)) for e in energies_not_none]
    logger.info(f'Maximum energy is {max(flat_rel_energy_array)} kcal mol-1')

    r1_flat = np.array([dists[0] for dists in dists_xyzs_energies_dict.keys()])
    r2_flat = np.array([dists[1] for dists in dists_xyzs_energies_dict.keys()])

    method_name = method.__name__
    if 'fbond' in scan_name:
        name = plot_name + f'_{method_name}_fbonds'
    elif 'bbond' in scan_name:
        name = plot_name + f'_{method_name}_bbonds'
    else:
        name = plot_name + f'_{method_name}'

    coeff_mat = polyfit2d(r1_flat, r2_flat, flat_rel_energy_array, order=order)

    saddle_points = poly2d_saddlepoints(coeff_mat)

    r1 = np.linspace(r1_flat[0], r1_flat[-1], n_points)
    r2 = np.linspace(r2_flat[0], r2_flat[-1], n_points)
    x, y = np.meshgrid(r1, r2)
    # do transpose so energy_grid[i,j] comes from r1[i] r2[j]
    energy_grid = polynomial.polyval2d(x, y, coeff_mat).transpose()

    saddle_points_in_range = []
    for point in saddle_points:
        if (min(r1_flat) < point[0] < max(r1_flat)) and (min(r2_flat) < point[1] < max(r2_flat)):
            saddle_points_in_range.append(point)
    if len(saddle_points_in_range) == 0:
        logger.error('No saddle points were found')
        plot_2dpes(r1_flat, r2_flat, flat_rel_energy_array, coeff_mat, name=name)
        return None
    elif len(saddle_points_in_range) == 1:
        min_energy_pathway = get_mep(r1, r2, energy_grid, saddle_points_in_range[0])
        if min_energy_pathway is None:
            plot_2dpes(r1_flat, r2_flat, flat_rel_energy_array, coeff_mat, name=name)
            return None
        r1_saddle, r2_saddle = saddle_points_in_range[0]
    else:
        logger.info(f'Found {len(saddle_points_in_range)} saddle points, finding the minimum energy pathway from each to reacs and prods')
        best_saddlepoint_output = best_saddlepoint(saddle_points_in_range, r1, r2, energy_grid)
        if best_saddlepoint_output is None:
            plot_2dpes(r1_flat, r2_flat, flat_rel_energy_array, coeff_mat, name=name)
            return None
        r1_saddle, r2_saddle, min_energy_pathway = best_saddlepoint_output

    logger.info(f'Found a saddle point at {r1_saddle:.3f}, {r2_saddle:.3f}')

    logger.info('Finding closest scan points to minimum energy pathway')
    mep_distances = []
    for coord in min_energy_pathway:
        mep_distances.append((r1[coord[0]], r2[coord[1]]))
    mep_xyzs = []
    for r1_dist, r2_dist in mep_distances:
        closest_dists = get_closest_point_dists_to_saddle(r1_dist, r2_dist, dists_xyzs_energies_dict.keys())
        coord_xyzs = dists_xyzs_energies_dict[closest_dists][0]
        if not coord_xyzs in mep_xyzs:
            mep_xyzs.append(coord_xyzs)

    plot_2dpes(r1_flat, r2_flat, flat_rel_energy_array,
               coeff_mat, mep_distances, name=name)

    make_reaction_animation(scan_name, mep_xyzs)

    logger.info('Getting the closest scan point to the analytic saddle point')

    closest_scan_point_dists = get_closest_point_dists_to_saddle(r1_saddle, r2_saddle, dists_xyzs_energies_dict.keys())
    xyzs_ts_guess = dists_xyzs_energies_dict[closest_scan_point_dists][0]

    return xyzs_ts_guess, (r1_saddle, r2_saddle)


def get_closest_point_dists_to_saddle(r1_saddle, r2_saddle, dists):
    """Finds the closest scan point to a pair of coordinates.

    Arguments:
        r1_saddle (float): point on the r1 axis
        r2_saddle (float): point on the r2 axis
        dists (list(tuple)): list of r1, r2 distances for points that have been scanned

    Returns:
        tuple: the closest scan point
    """
    closest_dist_to_saddle = 99999.9
    scan_dists_tuple = None

    for dist in dists:
        dist_to_saddle = np.linalg.norm(np.array(dist) - np.array([r1_saddle, r2_saddle]))
        if dist_to_saddle < closest_dist_to_saddle:
            closest_dist_to_saddle = dist_to_saddle
            scan_dists_tuple = dist

    return scan_dists_tuple





def replace_none(lst):
    """Replace Nones in a flat list with the closest preceding value

    Arguments:
        lst (list): list of value

    Yields:
        float: value from list
    """
    for item in lst:
        if item is not None:
            last = item
            break
    for item in lst:
        if item is None:
            yield last
        else:
            last = item
            yield item


def polyfit2d(x, y, z, order):
    """Takes x and y coordinates and their resultant z value, and creates a matrix where element i,j is the coefficient of the desired order polynomial x ** i * y ** j

    Arguments:
        x (np.array): flat array of x coordinates
        y (np.array): flat array of y coordinates
        z (np.array): flat array of z value at the corresponding x and y value
        order (int): max order of polynomial to work out

    Returns:
        np.array: matrix of polynomial coefficients
    """
    logger.info('Fitting 2D surface to polynomial in x and y')
    deg = np.array([order, order])
    vander = polynomial.polyvander2d(x, y, deg)
    # vander matrix is matrix where each row i deals with x=x[i] and y=y[i], and each item in the
    # row has value x ** m * y ** n with (m,n) = (0,0), (0,1), (0,2) ... (1,0), (1,1), (1,2) etc up to (order, order)
    coeff_mat, _, _, _ = np.linalg.lstsq(vander, z, rcond=None)
    return coeff_mat.reshape(deg + 1)
