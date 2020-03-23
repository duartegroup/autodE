import os
import numpy as np
from numpy.random import RandomState
from copy import deepcopy
from itertools import combinations
from scipy.optimize import minimize
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
    """If two stereocentres are bonded, rotate them randomly with respect to each other

    Arguments:
        species (autode.species.Species):
        atoms (list(autode.atoms.Atom)):
        theta (float): Rotation angle in radians
    """

    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    # Check on every pair of stereocenters
    for (atom_i, atom_j) in combinations(stereocentres, 2):
        if (atom_i, atom_j) in species.graph.edges:

            # Don't rotate if the bond connecting the centers is a π-bond
            if species.graph.edges[atom_i, atom_j]['pi'] is True:
                logger.info('Stereocenters were π bonded – not rotating')
                continue

            left_idxs, right_idxs = split_mol_across_bond(species.graph, bond=(atom_i, atom_j))

            # Rotate the left hand side randomly
            rot_axis = atoms[atom_i].coord - atoms[atom_j].coord
            [atoms[i].rotate(axis=rot_axis, theta=theta, origin=atoms[atom_i].coord) for i in left_idxs]

    return atoms


def get_non_random_atoms(species):
    """Get the atoms that won't be randomised in the conformer generation. Stereocentres and nearest neighbours"""
    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    non_rand_atoms = [i for (i, j) in species.graph.edges if i in stereocentres or j in stereocentres]
    non_rand_atoms += [j for (i, j) in species.graph.edges if i in stereocentres or j in stereocentres]

    return set(non_rand_atoms)


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
    atoms = get_atoms_rotated_stereocentres(species=species, atoms=deepcopy(species.atoms), theta=2*np.pi*rand.rand())

    # Add the distance constraints as fixed bonds
    d0 = get_ideal_bond_length_matrix(atoms=species.atoms, bonds=species.graph.edges())

    fixed_bonds = []
    if dist_consts is not None:
        for bond, length in dist_consts.items():
            d0[bond[0], bond[1]] = length
            d0[bond[1], bond[0]] = length
            fixed_bonds.append(bond)

    # Randomise coordinates
    non_rand_atom_indexes = get_non_random_atoms(species=species)
    [atom.translate(vec=rand.uniform(-1.0, 1.0, 3)) for i, atom in enumerate(atoms) if i not in non_rand_atom_indexes]

    logger.info('Minimising with BFGS')
    coords = get_coords_minimised_v(coords=np.array([atom.coord for atom in atoms]), bonds=species.graph.edges,
                                    k=10000, c=100, d0=d0, tol=species.n_atoms/5, fixed_bonds=fixed_bonds)

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    return atoms
