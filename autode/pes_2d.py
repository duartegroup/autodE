import numpy as np
from copy import deepcopy
from multiprocessing.pool import Pool
from autode.config import Config
from autode.constants import Constants
from autode.log import logger
from autode.calculation import Calculation
from autode.plotting import plot_2dpes
from autode.plotting import make_reaction_animation
from autode.transition_states.ts_guess import TSguess
from autode.exceptions import XYZsNotFound
from autode import mol_graphs


def get_ts_guess_2d(mol, product, active_bond1, active_bond2, n_steps, name, reaction_class, method, keywords,
                    delta_dist1=1.5, delta_dist2=1.5, active_bonds_not_scanned=None):
    """
    :param mol:
    :param product: single product molecule object
    :param active_bond1:
    :param active_bond2:
    :param n_steps:
        :param keywords (list) list of keywords required by an electronic structure method
    :param method: (object) electronic structure method
    :param reaction_class:
    :param products: list of product molecule objects
    :param name:
    :param delta_dist1:
    :param delta_dist2:
    :return:
    """
    logger.info(
        f'Getting TS guess from 2D relaxed potential energy scan, using active bonds {active_bond1} (delta distance = {delta_dist1:.3f}) and {active_bond2} (delta distance = {delta_dist2:.3f})')

    curr_dist1 = mol.distance_matrix[active_bond1[0], active_bond1[1]]
    curr_dist2 = mol.distance_matrix[active_bond2[0], active_bond2[1]]

    dist_grid1, dist_grid2 = np.meshgrid(np.linspace(curr_dist1, curr_dist1 + delta_dist1, n_steps),
                                         np.linspace(curr_dist2, curr_dist2 + delta_dist2, n_steps))

    # Create a grid of molecules and associated constrained optimisation calculations
    mol_grid = [[deepcopy(mol) for _ in range(n_steps)]
                for _ in range(n_steps)]

    # Perform a 1d scan in serial
    for n in range(n_steps):
        if n == 0:
            molecule = mol
        else:
            molecule = mol_grid[0][n-1]

        const_opt = Calculation(name=name + '_scan0_' + str(n), molecule=molecule, method=method, opt=True,
                                n_cores=Config.n_cores, distance_constraints={active_bond1: dist_grid1[0][n],
                                                                              active_bond2: dist_grid2[0][n]},
                                keywords=keywords)
        const_opt.run()
        # Set the new xyzs as those output from the calculation, and the previous if no xyzs could be found
        try:
            # Set the new xyzs of the molecule
            mol_grid[0][n].xyzs = const_opt.get_final_xyzs()
        except XYZsNotFound:
            mol_grid[0][n].xyzs = mol_grid[0][n-1].xyzs if n != 0 else mol.xyzs

        # Set the energy of the molecule. Can be None
        mol_grid[0][n].energy = const_opt.get_energy()

    # Execute the remaining set of optimisations in parallel
    for i in range(1, n_steps):

        if Config.n_cores <= n_steps:
            cores_per_process = 1
        else:
            cores_per_process = Config.n_cores//n_steps
            if Config.n_cores % n_steps != 0:
                logger.warning(
                    f'Not all cores will be used in the multiprocessing stage, for optimal core usage use a multiple of {n_steps} cores')

        calcs = [Calculation(name+'_scan'+str(i)+'_'+str(n), mol_grid[i-1][n], method, n_cores=cores_per_process, opt=True,
                             keywords=keywords, distance_constraints={active_bond1: dist_grid1[i][n],
                                                                      active_bond2: dist_grid2[i][n]})
                 for n in range(n_steps)]

        [calc.generate_input() for calc in calcs]
        if Config.n_cores <= n_steps:
            with Pool(processes=Config.n_cores) as pool:
                results = [pool.apply_async(execute_calc, (calc,))
                           for calc in calcs]
                [res.get(timeout=None) for res in results]
        else:
            with Pool(processes=n_steps) as pool:
                results = [pool.apply_async(execute_calc, (calc,))
                           for calc in calcs]
                [res.get(timeout=None) for res in results]
        [calc.set_output_file_lines() for calc in calcs]

        # Add attributes for molecules in the mol_grid
        for n in range(n_steps):
            calcs[n].terminated_normally = calcs[n].calculation_terminated_normally()
            # Set the new xyzs as those output from the calculation, and the previous if no xyzs could be found
            try:
                mol_grid[i][n].xyzs = calcs[n].get_final_xyzs()
            except XYZsNotFound:
                mol_grid[i][n].xyzs = deepcopy(mol_grid[i-1][n].xyzs)

            # Set the energy, this may be None
            mol_grid[i][n].energy = calcs[n].get_energy()

    # Populate the dictionary of distances, xyzs and energies – legacy
    dist_xyzs_energies = {}
    for n in range(n_steps):
        for m in range(n_steps):
            dist_xyzs_energies[(dist_grid1[n, m], dist_grid2[n, m])] = (
                mol_grid[n][m].xyzs, mol_grid[n][m].energy)

    # check product and TSGuess product graphs are isomorphic
    expected_prod_graphs = mol_graphs.get_separate_subgraphs(product.graph)
    logger.info('Checking products were made')
    products_made = False
    for row in mol_grid[::-1]:
        ts_product_graphs = [mol_graphs.make_graph(mol.xyzs, mol.n_atoms)
                             for mol in row[::-1]]
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
    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_2dpes_maximum_energy_xyzs(
        dist_xyzs_energies, scan_name=name, plot_name=mol.name + '_2dscan', method=method))

    mep_xyzs = [mol.xyzs]

    for i in range(n_steps):
        xyz_list = [mol.xyzs for mol in mol_grid[i]]
        energies_list = [mol.energy for mol in mol_grid[i]]
        energies_not_none = list(replace_none(energies_list))
        min_energy = min(energies_not_none)
        min_energy_index = energies_list.index(min_energy)
        min_xyz = xyz_list[min_energy_index]
        mep_xyzs.append(min_xyz)

    make_reaction_animation(name, mep_xyzs)

    active_bonds = [active_bond1, active_bond2] if active_bonds_not_scanned is None else [
        active_bond1, active_bond2] + active_bonds_not_scanned

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol,
                   active_bonds=active_bonds, reactant=mol, product=product)


def find_2dpes_maximum_energy_xyzs(dists_xyzs_energies_dict, scan_name, plot_name, method):
    """
    Find the first order saddle point on a 2D PES given a list of lists defined by their energy
    :param dists_xyzs_energies_dict: (dict) [value] = (xyzs, energy)
    :param scan_name: (str)
    :param plot_name: (str) name to use in the plot
    :return:
    """

    def poly2d_sationary_points(c_vec):

        a = np.array([[c_vec[3], 2.0 * c_vec[4]], [2.0 * c_vec[5], c_vec[3]]])
        b = np.array([-c_vec[2], -c_vec[1]])
        y_stat_point, x_stat_point = np.linalg.solve(a, b)

        return x_stat_point, y_stat_point

    logger.info('Finding saddle point in 2D PES')

    energies = [dists_xyzs_energies_dict[dists][1]
                for dists in dists_xyzs_energies_dict.keys()]
    # The energy lis may have None values in, so to perform the fitting replace with the closest float value
    energies_not_none = list(replace_none(energies))
    flat_rel_energy_array = [Constants.ha2kcalmol *
                             (e - min(energies_not_none)) for e in energies_not_none]
    logger.info(f'Maximum energy is {max(flat_rel_energy_array)} kcal mol-1')

    r1_flat = np.array([dists[0] for dists in dists_xyzs_energies_dict.keys()])
    r2_flat = np.array([dists[1] for dists in dists_xyzs_energies_dict.keys()])

    m = polyfit2d(r1_flat, r2_flat, flat_rel_energy_array)
    r1_saddle, r2_saddle = poly2d_sationary_points(m)
    logger.info(f'Found a saddle point at {r1_saddle}, {r2_saddle}')
    if r1_saddle < 0 or r2_saddle < 0:
        logger.error('2D surface has saddle points with negative distances!')

    method_name = method.__name__
    if 'fbond' in scan_name:
        name = plot_name + f'_{method_name}_fbonds'
    elif 'bbond' in scan_name:
        name = plot_name + f'_{method_name}_bbonds'
    else:
        name = plot_name + f'_{method_name}'

    logger.info(f'Plotting 2D scan and saving to {name}.png')
    plot_2dpes(r1_flat, r2_flat, flat_rel_energy_array, name=name)

    closest_scan_point_dists = get_closest_point_dists_to_saddle(
        r1_saddle, r2_saddle, dists_xyzs_energies_dict.keys())
    xyzs_ts_guess = dists_xyzs_energies_dict[closest_scan_point_dists][0]

    return xyzs_ts_guess


def get_closest_point_dists_to_saddle(r1_saddle, r2_saddle, dists):
    logger.info('Getting the closest scan point to the analytic saddle point')

    closest_dist_to_saddle = 99999.9
    scan_dists_tuple = None

    for dist in dists:
        dist_to_saddle = np.linalg.norm(
            np.array(dist) - np.array([r1_saddle, r2_saddle]))
        if dist_to_saddle < closest_dist_to_saddle:
            closest_dist_to_saddle = dist_to_saddle
            scan_dists_tuple = dist

    return scan_dists_tuple


def polyfit2d(x, y, z):  # order=2
    logger.info('Fitting 2D surface to 2nd order polynomial in x and y')
    # ncols = (order + 1) ** 2
    ij = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2)]
    g = np.zeros((x.size, len(ij)))
    # ij = itertools.product(range(order + 1), range(order + 1)))
    for k, (i, j) in enumerate(ij):
        # print(k, 'x order', i, 'y order', j)
        g[:, k] = x ** i * y ** j
    m, _, _, _ = np.linalg.lstsq(g, z, rcond=None)
    return m


def execute_calc(calc):
    return calc.execute_calculation()


def replace_none(lst):
    """
    Replace Nones in a flat list with the closest preceding value
    :param lst: (list)
    :return:
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
