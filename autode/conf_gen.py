from .log import logger
from .config import Config
from multiprocessing import Pool
from cconf_gen import do_md
from cconf_gen import v
import os
from scipy.optimize import minimize
from .bond_lengths import get_ideal_bond_length_matrix
from .input_output import xyzs2xyzfile
from .XTBio import run_xtb
from .XTBio import get_xtb_xyzs_energy


# def v(r, bonds, k, d0, c):
#
#     pot_e = 0
#
#     for i in range(int(len(r)/3)):
#         for j in range(int(len(r)/3)):
#             if i > j:
#
#                 d = np.sqrt((r[3*i] - r[3*j])**2 + (r[3*i+1] - r[3*j+1])**2 + (r[3*i+2] - r[3*j+2])**2)
#                 pot_e += c / d**4
#
#                 if (i, j) in bonds or (j, i) in bonds:
#                     pot_e += k * (d - d0[i, j]) ** 2
#
#     return pot_e


def get_coords_minimised_v(coords, bonds, k, c, d0, tol):
    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)
    init_coords = coords.reshape(3 * n_atoms, 1)
    res = minimize(v, x0=init_coords, args=(bonds, k, d0, c), method='BFGS', tol=tol)
    final_coords = res.x.reshape(n_atoms, 3)

    return final_coords


def simnal(name, xyzs, bonds, n, charge):
    """
        V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4
    :param name: (str)
    :param xyzs:
    :param bonds:
    :param n: (int) number of the simulated annealing calculation
    :param charge:
    :return:
    """

    n_steps = 100000
    temp = 10
    dt = 0.001
    k = 10000
    d0 = get_ideal_bond_length_matrix(xyzs, bonds)
    c = 100

    logger.info('Running high temp MD')
    coords = do_md(xyzs, bonds, n_steps, temp, dt, k, d0, c)

    logger.info('Minimising with BFGS')
    coords = get_coords_minimised_v(coords, bonds, k, c, d0, tol=len(xyzs)/5)
    xyzs = [[xyzs[i][0]] + coords[i].tolist() for i in range(len(xyzs))]

    # Optimise the rough structure with XTB
    xyz_filename = name + '_simanl_' + str(n) + '.xyz'
    xyzs2xyzfile(xyzs, filename=xyz_filename)

    xtb_out_lines = run_xtb(xyz_filename, opt=True, charge=charge, n_cores=1)
    xyzs, _ = get_xtb_xyzs_energy(xtb_out_lines)

    return xyzs


def gen_simanl_conf_xyzs(name, init_xyzs, bond_list, charge, n_simanls=20):
    """
    Generate conformer xyzs using the cconf_gen Cython code, which is compiled when setup.py install is run.

    :param name: (str) name of the molecule to run. needed for XTB filenames
    :param init_xyzs: (list(list)) xyz list to work from
    :param bond_list: (list(tuple)) list of bond indices
    :param charge: (int) charge on the molecule for an XTB optimisation
    :param n_simanls: (int) number of simulated anneling steps to do
    :return:
    """
    logger.info('Doing simulated annealing with a harmonic+repulsion FF')

    logger.info('Splitting calculation into {} threads'.format(Config.n_cores))
    with Pool(processes=Config.n_cores) as pool:
        results = [pool.apply_async(simnal, (name, init_xyzs, bond_list, i, charge)) for i in range(n_simanls)]
        conf_xyzs = [res.get(timeout=None) for res in results]

    return conf_xyzs
