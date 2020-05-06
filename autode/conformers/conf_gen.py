from copy import deepcopy
from itertools import combinations
import networkx as nx
import numpy as np
import os
from scipy.optimize import minimize
from time import time
from cconf_gen import v
from cconf_gen import dvdr
from autode.atoms import get_vdw_radius
from autode.atoms import get_atomic_weight
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.bond_lengths import get_avg_bond_length
from autode.config import Config
from autode.input_output import xyz_file_to_atoms
from autode.input_output import atoms_to_xyz_file
from autode.log import logger
from autode.mol_graphs import split_mol_across_bond
from autode.exceptions import CannotSplitAcrossBond

from autode.atoms import Atom


def get_coords_minimised_err(coords, species, dist_consts):

    os.environ['OMP_NUM_THREADS'] = str(1)

    init_coords = coords.reshape(3 * species.n_atoms)
    init_coords = np.random.rand(3 * species.n_atoms)
    dist_dict = {}
    for i, j in species.graph.edges():
        if (i, j) in dist_consts.keys():
            continue
        ideal_length = get_avg_bond_length(species.atoms[i].label, species.atoms[j].label)
        upper_bound_2 = (ideal_length * 1.05)**2
        lower_bound_2 = (ideal_length * 0.95)**2
        dist_dict[(i, j)] = (upper_bound_2, lower_bound_2)
    for (i, j), distance in dist_consts.items():
        dist_dict[(i, j)] = (distance**2, distance**2)
    res = minimize(dist_erf, x0=init_coords, args=(dist_dict), method='CG')

    return res.x.reshape(species.n_atoms, 3)


def get_atoms_from_generated_file(species, xyz_filename):
    """Get the atoms from a previously generated  .xyz file, if the atoms match"""

    if not os.path.exists(xyz_filename):
        return None

    atoms = xyz_file_to_atoms(filename=xyz_filename)

    # Ensure the xyz file has the correct atoms
    all_atoms_match = all(atoms[i].label == species.atoms[i].label for i in range(species.n_atoms))

    if len(atoms) == species.n_atoms and all_atoms_match:
        logger.info('Conformer has already been generated')
        return atoms

    return None


def get_triangle_inequality_limits(species, dist_consts):
    upper_limit = np.zeros(shape=(species.n_atoms, species.n_atoms))
    lower_limit = np.zeros(shape=(species.n_atoms, species.n_atoms))
    weighted_digraph = nx.MultiDiGraph()
    for i, j in species.graph.edges():
        if (i, j) in dist_consts.keys():
            continue
        ideal_length = get_avg_bond_length(species.atoms[i].label, species.atoms[j].label)
        weighted_digraph.add_edge(i, j, weight=(ideal_length * 1.05))
        weighted_digraph.add_edge(i+species.n_atoms, j+species.n_atoms, weight=(ideal_length * 1.05))
        weighted_digraph.add_edge(j, i, weight=(ideal_length * 1.05))
        weighted_digraph.add_edge(j+species.n_atoms, i+species.n_atoms, weight=(ideal_length * 1.05))
        weighted_digraph.add_edge(i, j+species.n_atoms, weight=-1*(ideal_length * 0.95))
        weighted_digraph.add_edge(j, i+species.n_atoms, weight=-1*(ideal_length * 0.95))
    for (i, j), distance in dist_consts.items():
        weighted_digraph.add_edge(i, j, weight=distance)
        weighted_digraph.add_edge(i+species.n_atoms, j+species.n_atoms, weight=distance)
        weighted_digraph.add_edge(j, i, weight=distance)
        weighted_digraph.add_edge(j+species.n_atoms, i+species.n_atoms, weight=distance)
        weighted_digraph.add_edge(i, j+species.n_atoms, weight=-1*distance)
        weighted_digraph.add_edge(j, i+species.n_atoms, weight=-1*distance)

    distance_dict = nx.floyd_warshall(weighted_digraph)
    for atom_i in range(species.n_atoms):
        atom_i_dist_dict = dict(distance_dict[atom_i])
        for atom_j in range(species.n_atoms):
            if atom_i != atom_j:
                upper_limit[atom_i, atom_j] = atom_i_dist_dict[atom_j]
                if atom_i_dist_dict[atom_j+species.n_atoms] < 0:
                    lower_limit[atom_i, atom_j] = -1*atom_i_dist_dict[atom_j+species.n_atoms]
                else:
                    lower_limit[atom_i, atom_j] = 1

    return upper_limit, lower_limit


def dist_erf(coords, dist_dict):
    error = 0
    for (i, j), (upper_bound_2, lower_bound_2) in dist_dict.items():
        bond_vec = coords[3*i:3*(i+1)] - coords[3*j:3*(j+1)]
        dist_2 = np.dot(bond_vec, bond_vec)
        upper_err = (dist_2 - upper_bound_2)/(1+upper_bound_2)
        lower_err = (lower_bound_2 - dist_2)/(1+dist_2)
        if upper_err > 0:
            error += upper_err**2
        if lower_err > 0:
            error += lower_err**2
    for i, j in combinations(range(int(len(coords)/3)), 2):
        if (i, j) in dist_dict.keys():
            continue
        bond_vec = coords[3*i:3*(i+1)] - coords[3*j:3*(j+1)]
        dist_2 = np.dot(bond_vec, bond_vec)
        err = (20 - dist_2)/(1+dist_2)
        if err > 0:
            error += err**2
    return error


def get_simanl_atoms(species, dist_consts={}, conf_n=0):
    """
    Arguments:
        species (autode.species.Species): Species, Molecule, TSguess, TS
        dist_consts (dict): Key = tuple of atom indexes, Value = distance
        conf_n (int): Number of this conformer generated

    Returns:
        (np.ndarray): Coordinates of the generated conformer
    """
    xyz_filename = f'{species.name}_conf{conf_n}_siman.xyz'

    # saved_atoms = get_atoms_from_generated_file(species, xyz_filename)
    # if saved_atoms is not None:
    #     return saved_atoms

    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = np.random.RandomState()

    logger.info('Minimising species...')
    st = time()

    upper_limit, lower_limit = get_triangle_inequality_limits(species, dist_consts)

    dist_matrix = np.zeros(shape=(species.n_atoms, species.n_atoms))
    for i, j in combinations(range(species.n_atoms), 2):
        distance = (upper_limit[i, j] - lower_limit[i, j]) * (rand.rand()) + lower_limit[i, j]
        dist_matrix[i, j] = distance**2
        dist_matrix[j, i] = distance**2

    m = np.zeros(shape=(1, species.n_atoms))
    W = np.zeros(shape=(species.n_atoms, species.n_atoms))
    for i, atom in enumerate(species.atoms):
        weight = get_atomic_weight(atom.label)
        m[0, i] = weight
        W[i, i] = weight

    left_matrix = np.identity(species.n_atoms) - (np.matmul(np.ones(shape=(species.n_atoms, 1)), m))/np.sum(m)
    right_matrix = np.identity(species.n_atoms) - (np.matmul(np.transpose(m), np.ones(shape=(1, species.n_atoms))))/np.sum(m)

    A = -0.5 * np.matmul(left_matrix, np.matmul(dist_matrix, right_matrix))
    B = np.matmul(W, np.matmul(A, W))

    eigenvals, eigenvecs = np.linalg.eig(B)

    Y = np.zeros(shape=(species.n_atoms, 3))
    sorted_eigenvals = sorted(eigenvals, reverse=True)
    for i in range(3):
        index = eigenvals.tolist().index(sorted_eigenvals[i])
        eigenval = sorted_eigenvals[i]
        for j in range(species.n_atoms):
            Y[j, i] = np.sqrt(eigenval) * eigenvecs[j, index]
    X = np.matmul(np.linalg.inv(W), Y)

    coords = get_coords_minimised_err(X, species, dist_consts)
    logger.info(f'                  ... ({time()-st:.3f} s)')

    coords = np.array([[-1.5, -1.5, 0],
                       [-1, 0, 0],
                       [-1.5, 1.5, 0],
                       [1.5, -1.5, 0],
                       [1.5, 1.5, 0],
                       [1, 0, 0]])

    vec1 = (coords[2] - coords[0])/np.linalg.norm((coords[2] - coords[0]))
    vec2 = (coords[3] - coords[0])/np.linalg.norm((coords[3] - coords[0]))
    vec3 = (coords[4] - coords[0])/np.linalg.norm((coords[4] - coords[0]))
    print(np.dot(vec1, np.cross(vec2, vec3)))

    atoms = []
    for i in range(species.n_atoms):
        x, y, z = X[i]
        atoms.append(Atom(species.atoms[i].label, x, y, z))

    atoms_to_xyz_file(atoms=atoms, filename=xyz_filename)

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    # # Print an xyz file so rerunning will read the file
    atoms_to_xyz_file(atoms=atoms, filename=f'minimised_{xyz_filename}')

    exit()

    # return atoms
