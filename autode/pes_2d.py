import numpy as np
from copy import deepcopy
from multiprocessing.pool import Pool
from autode.config import Config
from autode.constants import Constants
from autode.log import logger
from autode.calculation import Calculation
from autode.wrappers.ORCA import ORCA
from autode.wrappers.XTB import XTB
from autode.plotting import plot_2dpes
from autode.ts_guess import TSguess


def find_2dpes_maximum_energy_xyzs(dists_xyzs_energies_dict):
    """
    Find the first order saddle point on a 2D PES given a list of lists defined by their energy
    :param dists_xyzs_energies_dict: (dict) [value] = (xyzs, energy)
    :return:
    """

    def poly2d_sationary_points(c_vec):

        a = np.array([[c_vec[3], 2.0 * c_vec[4]], [2.0 * c_vec[5], c_vec[3]]])
        b = np.array([-c_vec[2], -c_vec[1]])
        y_stat_point, x_stat_point = np.linalg.solve(a, b)

        return x_stat_point, y_stat_point

    logger.info('Finding saddle point in 2D PES')

    energies = [dists_xyzs_energies_dict[dists][1] for dists in dists_xyzs_energies_dict.keys()]

    r1_flat = np.array([dists[0] for dists in dists_xyzs_energies_dict.keys()])
    r2_flat = np.array([dists[1] for dists in dists_xyzs_energies_dict.keys()])

    flat_rel_energy_array = Constants.ha2kcalmol * (np.array(energies) - min(energies))

    m = polyfit2d(r1_flat, r2_flat, flat_rel_energy_array)
    r1_saddle, r2_saddle = poly2d_sationary_points(m)
    logger.info('Found a saddle point at {}, {}'.format(r1_saddle, r2_saddle))
    plot_2dpes(r1_flat, r2_flat, flat_rel_energy_array)

    closest_scan_point_dists = get_closest_point_dists_to_saddle(r1_saddle, r2_saddle, dists_xyzs_energies_dict.keys())
    xyzs_ts_guess = dists_xyzs_energies_dict[closest_scan_point_dists][0]

    return xyzs_ts_guess


def get_closest_point_dists_to_saddle(r1_saddle, r2_saddle, dists):
    logger.info('Getting the closest scan point to the analytic saddle point')

    closest_dist_to_saddle = 99999.9
    scan_dists_tuple = None

    for dist in dists:
        dist_to_saddle = np.linalg.norm(np.array(dist) - np.array([r1_saddle, r2_saddle]))
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


def get_est_ts_guess_2d(mol, active_bond1, active_bond2, n_steps, reaction_class, orca_keywords, name='2d',
                        delta_dist1=1.5, delta_dist2=1.5):
    logger.info('Getting TS guess from 2D ORCA relaxed potential energy scan')

    curr_dist1 = mol.distance_matrix[active_bond1[0], active_bond1[1]]
    curr_dist2 = mol.distance_matrix[active_bond2[0], active_bond2[1]]

    scan = Calculation(name=name + '_2dscan', molecule=mol, method=ORCA, keywords=orca_keywords,
                       n_cores=Config.n_cores, max_core_mb=Config.max_core, scan_ids=active_bond1,
                       curr_dist1=curr_dist1, final_dist1=curr_dist1 + delta_dist1, opt=True, scan_ids2=active_bond2,
                       curr_dist2=curr_dist2, final_dist2=curr_dist2 + delta_dist2, n_steps=n_steps)
    scan.run()

    dists_xyzs_energies = scan.get_scan_values_xyzs_energies()
    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_2dpes_maximum_energy_xyzs(dists_xyzs_energies))

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol,
                   active_bonds=[active_bond1, active_bond2])


def get_xtb_ts_guess_2d(mol, active_bond1, active_bond2, n_steps, reaction_class, name, delta_dist1=1.5,
                        delta_dist2=1.5):
    """

    :param mol:
    :param active_bond1:
    :param active_bond2:
    :param n_steps:
    :param reaction_class:
    :param name:
    :param delta_dist1:
    :param delta_dist2:
    :return:
    """
    logger.info('Getting TS guess from 2D XTB relaxed potential energy scan')

    curr_dist1 = mol.distance_matrix[active_bond1[0], active_bond1[1]]
    curr_dist2 = mol.distance_matrix[active_bond2[0], active_bond2[1]]

    dist_grid1, dist_grid2 = np.meshgrid(np.linspace(curr_dist1, curr_dist1 + delta_dist1, n_steps),
                                         np.linspace(curr_dist2, curr_dist2 + delta_dist2, n_steps))

    # Create a grid of molecules and associated constrained optimisation calculations
    mol_grid = [[deepcopy(mol) for _ in range(n_steps)] for _ in range(n_steps)]

    # Perform a 1d scan in serial
    for n in range(n_steps):
        if n == 0:
            molecule = mol
        else:
            molecule = mol_grid[0][n-1]

        const_opt = Calculation(name=name + '_scan0_' + str(n), molecule=molecule, method=XTB, opt=True,
                                n_cores=Config.n_cores, distance_constraints={active_bond1: dist_grid1[0][n],
                                                                              active_bond2: dist_grid2[0][n]})
        const_opt.run()
        # const_opt.run()
        mol_grid[0][n].xyzs = const_opt.get_final_xyzs()    # Set the new xyzs of the molecule
        mol_grid[0][n].energy = const_opt.get_energy()      # Set the energy of the molecule

    # Execute the remaining set of optimisations in parallel
    for i in range(1, n_steps):

        calcs = [Calculation(name+'_scan'+str(i)+'_'+str(n), mol_grid[i-1][n], XTB, n_cores=1, opt=True,
                             distance_constraints={active_bond1: dist_grid1[i][n], active_bond2: dist_grid2[i][n]})
                 for n in range(n_steps)]

        [calc.generate_input() for calc in calcs]
        with Pool(processes=Config.n_cores) as pool:
            results = [pool.apply_async(execute_calc, (calc,)) for calc in calcs]
            [res.get(timeout=None) for res in results]
        [calc.set_output_file_lines() for calc in calcs]

        # Add attributes for molecules in the mol_grid
        for n in range(n_steps):
            calcs[n].terminated_normally = calcs[n].calculation_terminated_normally()
            mol_grid[i][n].xyzs = calcs[n].get_final_xyzs()
            mol_grid[i][n].energy = calcs[n].get_energy()

    # Populate the dictionary of distances, xyzs and energies – legacy
    dist_xyzs_energies = {}
    for n in range(n_steps):
        for m in range(n_steps):
            dist_xyzs_energies[(dist_grid1[n, m], dist_grid2[n, m])] = (mol_grid[n][m].xyzs, mol_grid[n][m].energy)

    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_2dpes_maximum_energy_xyzs(dist_xyzs_energies))

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol,
                   active_bonds=[active_bond1, active_bond2])


def execute_calc(calc):
    return calc.execute_calculation()
