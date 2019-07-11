import numpy as np
from .config import Config
from .constants import Constants
from .log import logger
from .input_output import xyzs2xyzfile
from .XTBio import run_xtb
from .XTBio import get_xtb_scan_xyzs_energies
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca
from .ORCAio import get_orca_scan_values_xyzs_energies
from .plotting import plot_2dpes
from .ts_guess import TSguess


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


def get_orca_ts_guess_2d(mol, bond_ids, curr_dist1, final_dist1, curr_dist2, final_dist2, reaction_class,
                         n_steps=7, name='2d', orca_keywords=Config.scan_keywords):
    logger.info('Getting TS guess from 2D ORCA relaxed potential energy scan')

    reac_scan_inp_filename = name + '_orca_scan.inp'
    gen_orca_inp(reac_scan_inp_filename, orca_keywords, mol.xyzs, mol.charge, mol.mult,
                 mol.solvent, Config.n_cores, scan_ids=bond_ids[0], curr_dist1=curr_dist1, final_dist1=final_dist1,
                 curr_dist2=curr_dist2, final_dist2=final_dist2, n_steps=n_steps, scan_ids2=bond_ids[1])

    orca_out_lines = run_orca(reac_scan_inp_filename, out_filename=reac_scan_inp_filename.replace('.inp', '.out'))

    dists_xyzs_energies = get_orca_scan_values_xyzs_energies(orca_out_lines, scan_2d=True)
    ts_guess_xyzs = find_2dpes_maximum_energy_xyzs(dists_xyzs_energies)

    return TSguess(name=name, reaction_class=reaction_class, xyzs=ts_guess_xyzs, solvent=mol.solvent,
                   charge=mol.charge, mult=mol.mult, bonds_to_add=bond_ids)


def get_xtb_ts_guess_2d(mol, bbond_atom_ids_and_dists, reaction_class, max_bond_dist_add=1.5, n_steps=15):
    logger.info('Getting TS guess from 2D XTB relaxed potential energy scan')

    reac_xyz_filename = xyzs2xyzfile(mol.xyzs, basename=mol.name)
    dists_xyzs_energies, xyzs, energies = {}, [], []

    curr_bond_dist1, curr_bond_dist2 = list(bbond_atom_ids_and_dists.values())
    final_dist = curr_bond_dist1 + max_bond_dist_add

    bond_ids1, bond_ids2 = list(bbond_atom_ids_and_dists.keys())
    for dist_constraint in np.linspace(curr_bond_dist1, final_dist, n_steps):
        run_xtb(reac_xyz_filename, charge=mol.charge, scan_ids=bond_ids2, solvent=mol.solvent,
                n_steps=n_steps, bond_constraints={bond_ids1: dist_constraint},
                out_filename=str(dist_constraint) + '_scan_xtb.out')
        dist2s_xyzs_energies = get_xtb_scan_xyzs_energies(values=np.linspace(curr_bond_dist1, final_dist, n_steps))

        for dist2 in dist2s_xyzs_energies.keys():
            dists_xyzs_energies[(dist_constraint, dist2)] = dist2s_xyzs_energies[dist2]

    ts_guess_xyzs = find_2dpes_maximum_energy_xyzs(dists_xyzs_energies)

    return TSguess(reaction_class=reaction_class, xyzs=ts_guess_xyzs, solvent=mol.solvent,
                   charge=mol.charge, mult=mol.mult, bonds_to_add=[bond_ids1, bond_ids2])
