import os
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.log import logger
from autode.config import Config
from autode.geom import xyz2coord
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix
from autode.mol_graphs import split_mol_across_bond
from multiprocessing import Pool
from cconf_gen import do_md
from cconf_gen import v


def get_coords_minimised_v(coords, bonds, k, c, d0, tol, fixed_bonds):
    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)
    init_coords = coords.reshape(3 * n_atoms, 1)
    res = minimize(v, x0=init_coords, args=(bonds, k, d0, c, fixed_bonds), method='BFGS', tol=tol)
    final_coords = res.x.reshape(n_atoms, 3)

    return final_coords


def simanl(xyzs, bonds, dist_consts, non_random_atoms, stereocentres):
    """V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4

    Arguments:
        xyzs (list(list)): e.g. [['C', 0.0, 0.0, 0.0], ...]
        bonds (list(tuples)): defining which atoms are bonded together
        dist_consts (dict): keys = tuple of atom ids for a bond to be kept at fixed length, value = length to be fixed at
        non_random_atoms (list): atoms that must not be randomly placed, to keep stereochem
        stereocentres (list): list of stereocentres

    Returns:
        list(list): e.g. [['C', 0.0, 0.0, 0.0], ...]
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

    graph = nx.Graph()
    for bond in bonds:
        graph.add_edge(*bond)

    coords = xyz2coord(xyzs)

    # if two stereocentres are bonded, rotate them randomly wrt each other
    if stereocentres is not None:
        for atom1 in stereocentres:
            for atom2 in stereocentres:
                if atom1 < atom2:
                    if (atom1, atom2) in bonds or (atom2, atom1) in bonds:
                        theta = np.random.random_sample() * np.pi * 2
                        split_atoms = split_mol_across_bond(graph, [(atom1, atom2)])
                        if atom1 in split_atoms[0]:
                            atoms_to_rot = split_atoms[0]
                        else:
                            atoms_to_rot = split_atoms[1]
                        coords = coords - coords[atom1]
                        rot_axis = coords[atom1] - coords[atom2]
                        rot_matrix = calc_rotation_matrix(rot_axis, theta)
                        for atom in atoms_to_rot:
                            coords[atom] = np.matmul(rot_matrix, coords[atom])

        xyzs = coords2xyzs(coords, xyzs)

    logger.info('Running high temp MD')
    coords = do_md(xyzs, bonds, n_steps, temp, dt, k, d0, c, fixed_bonds, non_random_atoms)

    logger.info('Minimising with BFGS')
    coords = get_coords_minimised_v(coords, bonds, k, c, d0, len(xyzs)/5, fixed_bonds)
    xyzs = [[xyzs[i][0]] + coords[i].tolist() for i in range(len(xyzs))]

    return xyzs


def gen_simanl_conf_xyzs(name, init_xyzs, bond_list, stereocentres, dist_consts={}, n_simanls=40):
    """Generate conformer xyzs using the cconf_gen Cython code, which is compiled when setup.py install is run.

    Arguments:
        name (str): name of the molecule to run, needed to check for existing confs
        init_xyzs (list(list)): e.g. [['C', 0.0, 0.0, 0.0], ...]
        bond_list (list(tuple)): defining which atoms are bonded together
        stereocentres (list): list of stereocentres

    Keyword Arguments:
        dist_consts (dict): keys = tuple of atom ids for a bond to be kept at fixed length, value = length to be fixed at (default: {{}})
        n_simanls (int): number of simulated anneling steps to do (default: {40})

    Returns:
        list(list(list)): list of n_simanls xyzs
    """
    logger.info('Doing simulated annealing with a harmonic + repulsion force field')

    important_stereoatoms = set()
    if stereocentres is not None:
        for stereocentre in stereocentres:
            important_atoms = set()
            for bond in bond_list:
                if stereocentre in bond:
                    important_atoms.add(bond[0])
                    important_atoms.add(bond[1])
            for atom1 in important_atoms:
                for atom2 in important_atoms:
                    if atom1 > atom2:
                        coord1 = np.asarray(init_xyzs[atom1][1:])
                        coord2 = np.asarray(init_xyzs[atom2][1:])
                        bond_length = np.linalg.norm(coord1 - coord2)
                        dist_consts[(atom1, atom2)] = bond_length
            important_stereoatoms.update(important_atoms)

    non_random_atoms = sorted(important_stereoatoms)

    if n_simanls == 1:
        conf_xyzs = simanl(xyzs=init_xyzs, bonds=bond_list, dist_consts=dist_consts,
                           non_random_atoms=non_random_atoms, stereocentres=stereocentres)
        return [conf_xyzs]

    all_conf_xyzs = []

    logger.info('Looking for previously generated conformers')

    no_conf = []
    for i in range(n_simanls):
        xyz_file_name_start = f'{name}_conf{i}'
        has_conf = False
        for filename in os.listdir(os.getcwd()):
            if filename.startswith(xyz_file_name_start) and filename.endswith('.xyz'):
                xyzs = []
                with open(filename, 'r') as file:
                    for line_no, line in enumerate(file):
                        if line_no > 1:
                            atom_label, x, y, z = line.split()
                            xyzs.append([atom_label, float(x), float(y), float(z)])
                all_conf_xyzs.append(xyzs)
                has_conf = True
                break
        if not has_conf:
            no_conf.append(i)

    logger.info(f'Found {len(all_conf_xyzs)} previously generated conformers')

    simanls_left = n_simanls - len(all_conf_xyzs)

    logger.info(f'Have {simanls_left} conformers left to generate')

    if simanls_left > 0:
        logger.info(f'Splitting calculation into {Config.n_cores} threads')
        with Pool(processes=Config.n_cores) as pool:
            results = [pool.apply_async(simanl, (init_xyzs, bond_list, dist_consts, non_random_atoms, stereocentres))
                       for i in range(simanls_left)]
            conf_xyzs = [res.get(timeout=None) for res in results]

        for i in range(simanls_left):
            all_conf_xyzs.insert(no_conf[i], conf_xyzs[i])

    return all_conf_xyzs
