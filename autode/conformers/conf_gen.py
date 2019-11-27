import os
import numpy as np
from scipy.optimize import minimize
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.log import logger
from autode.config import Config
from multiprocessing import Pool
from cconf_gen import do_md
from cconf_gen import v


def get_coords_minimised_v(coords, bonds, k, c, d0, tol, fixed_bonds):
    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)
    init_coords = coords.reshape(3 * n_atoms, 1)
    res = minimize(v, x0=init_coords, args=(
        bonds, k, d0, c, fixed_bonds), method='BFGS', tol=tol)
    final_coords = res.x.reshape(n_atoms, 3)

    return final_coords


def simanl(xyzs, bonds, dist_consts):
    """
        V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4
    :param name: (str)
    :param xyzs:
    :param bonds:
    :param n: (int) number of the simulated annealing calculation
    :param charge:
    :return:
    """

    np.random.seed()

    n_steps = 100000
    temp = 10
    dt = 0.001
    k = 10000
    d0 = get_ideal_bond_length_matrix(xyzs, bonds)
    c = 100
    fixed_bonds = []
    if dist_consts is not None:
        for bond, length in dist_consts.items():
            d0[bond[0], bond[1]] = length
            d0[bond[1], bond[0]] = length
            fixed_bonds.append(bond)

    logger.info('Running high temp MD')
    coords = do_md(xyzs, bonds, n_steps, temp, dt, k, d0, c, fixed_bonds)

    logger.info('Minimising with BFGS')
    coords = get_coords_minimised_v(
        coords, bonds, k, c, d0, len(xyzs)/5, fixed_bonds)
    xyzs = [[xyzs[i][0]] + coords[i].tolist() for i in range(len(xyzs))]

    return xyzs


def gen_simanl_conf_xyzs(name, init_xyzs, bond_list, charge, dist_consts=None, n_simanls=40):
    """
    Generate conformer xyzs using the cconf_gen Cython code, which is compiled when setup.py install is run.

    :param name: (str) name of the molecule to run. needed for XTB filenames
    :param init_xyzs: (list(list)) xyz list to work from
    :param bond_list: (list(tuple)) list of bond indices
    :param charge: (int) charge on the molecule for an XTB optimisation
    :param n_simanls: (int) number of simulated anneling steps to do
    :return:
    """
    logger.info(
        'Doing simulated annealing with a harmonic + repulsion force field')

    if n_simanls == 1:
        conf_xyzs = simanl(xyzs=init_xyzs, bonds=bond_list,
                           dist_consts=dist_consts)
        return [conf_xyzs]

    logger.info(f'Splitting calculation into {Config.n_cores} threads')
    with Pool(processes=Config.n_cores) as pool:
        results = [pool.apply_async(simanl, (init_xyzs, bond_list, dist_consts))
                   for i in range(n_simanls)]
        conf_xyzs = [res.get(timeout=None) for res in results]

    good_conf_xyzs = []

    for xyzs in conf_xyzs:
        good_xyz = True
        for xyz in xyzs:
            for item in xyz:
                if str(item) == 'nan':
                    good_xyz = False
                    break
            if not good_xyz:
                break
        if good_xyz:
            good_conf_xyzs.append(xyzs)

    return good_conf_xyzs
