import os
import numpy as np
from numpy.random import RandomState
from copy import deepcopy
from itertools import combinations
from scipy.optimize import minimize
from autode.utils import requires_graph
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.input_output import xyz_file_to_atoms
from autode.log import logger
from autode.config import Config
from autode.mol_graphs import split_mol_across_bond
from multiprocessing import Pool
from cconf_gen import v


def get_coords_minimised_v(coords, bonds, k, c, d0, tol, fixed_bonds):
    # TODO rewrite in Cython for speeed

    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)
    init_coords = coords.reshape(3 * n_atoms, 1)
    res = minimize(v, x0=init_coords, args=(bonds, k, d0, c, fixed_bonds), method='BFGS', tol=tol)

    return res.x.reshape(n_atoms, 3)


def get_atoms_rotated_stereocentres(species, atoms, theta):
    """If two stereocentres are bonded, rotate them randomly wrt each other

    Arguments:
        species (autode.species.Species):
        atoms (list(autode.atoms.Atom)):
        theta (float): Rotation angle in radians
    """
    # TODO check on π bonds and don't rotate

    if species.stereocentres is not None:
        for (atom1, atom2) in combinations(species.stereocentres, 2):
            if (atom1, atom2) in species.graph.edges or (atom2, atom1) in species.graph.edges:
                left_idxs, right_idxs = split_mol_across_bond(species.graph, bond=(atom1, atom2))

                # Rotate the left hand side randomly
                rot_axis = atoms[atom1].coord - atoms[atom2].coord
                [atoms[i].rotate(axis=rot_axis, theta=theta, origin=atoms[atom1].coord) for i in left_idxs]

    return atoms


def get_non_random_atoms(species):
    """Get the atoms that won't be randomised in the conformer generation. Stereocentres and nearest neighbours"""

    non_rand_atoms = [i for (i, j) in species.graph.edges if i in species.stereocentres or j in species.stereocentres]
    non_rand_atoms += [j for (i, j) in species.graph.edges if i in species.stereocentres or j in species.stereocentres]

    return set(non_rand_atoms)


@requires_graph()
def get_simanl_atoms(species, dist_consts=None, conf_n=0):
    """V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4

    Arguments:
        species (autode.species.Species): Species, Molecule, TSguess, TS
        dist_consts (dict): Key = tuple of atom indexes, Value = distance
        conf_n (int): Number of this conformer generated

    Returns:
        (np.ndarray): Coordinates of the generated conformer
    """
    xyz_filename = f'{species.name}_conf{conf_n}.xyz'

    for filename in os.listdir(os.getcwd()):
        if filename == xyz_filename:
            logger.info('Conformer has already been generated')
            return xyz_file_to_atoms(filename=filename)

    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = RandomState()
    atoms = get_atoms_rotated_stereocentres(species=species, atoms=deepcopy(species.atoms), theta=2*np.pi*rand.random())

    # Add the distance constraints as fixed bonds
    d0 = get_ideal_bond_length_matrix(atoms=species.atoms, bonds=species.graph.edges())

    fixed_bonds = []
    if dist_consts is not None:
        for bond, length in dist_consts.items():
            d0[bond[0], bond[1]] = length
            d0[bond[1], bond[0]] = length
            fixed_bonds.append(bond)

    # Randomise coordinates
    [atom.translate(vec=rand.uniform(-1.0, 1.0, 3)) for atom in atoms]

    logger.info('Minimising with BFGS')
    coords = get_coords_minimised_v(coords=np.array([atom.coord for atom in atoms]), bonds=species.graph.edges,
                                    k=10000, c=100, d0=d0, tol=species.n_atoms/5, fixed_bonds=fixed_bonds)

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    return atoms


def get_simanl_conformers(name, init_xyzs, bond_list, stereocentres, dist_consts={}, n_simanls=40):
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
        conf_xyzs = simanl(name=name, xyzs=init_xyzs, bonds=bond_list, dist_consts=dist_consts, non_random_atoms=non_random_atoms, stereocentres=stereocentres, n=0)
        return [conf_xyzs]

    logger.info(f'Generating {n_simanls} conformers')

    logger.info(f'Splitting calculation into {Config.n_cores} threads')
    with Pool(processes=Config.n_cores) as pool:
        results = [pool.apply_async(simanl, (name, init_xyzs, bond_list, dist_consts, non_random_atoms, stereocentres, i))
                   for i in range(n_simanls)]
        conf_xyzs = [res.get(timeout=None) for res in results]

    return conf_xyzs
