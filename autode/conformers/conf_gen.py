from copy import deepcopy
from itertools import combinations
import numpy as np
import os
from scipy.optimize import minimize
from time import time
from cconf_gen import v
from cconf_gen import dvdr
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.config import Config
from autode.input_output import xyz_file_to_atoms
from autode.input_output import atoms_to_xyz_file
from autode.log import logger
from autode.geom import are_coords_reasonable
from autode.mol_graphs import split_mol_across_bond
from autode.exceptions import CannotSplitAcrossBond
from autode.exceptions import NoMolecularGraph


def get_bond_matrix(n_atoms, bonds, fixed_bonds):
    """
    Populate a bond matrix with 1 if i, j are bonded, 2 if i, j are bonded and fixed and 0 otherwise. Can support
    a partial structure with bonds to atoms that don't (yet) exist

    Args:
        n_atoms (int):
        bonds (list(tuple)):
        fixed_bonds (list(tuple)):

    Returns:
        (np.ndarray): n_atoms x n_atoms
    """

    bond_matrix = np.zeros((n_atoms, n_atoms), dtype=np.intc)

    for i, j in bonds:
        if i < n_atoms and j < n_atoms:
            bond_matrix[i, j] = 1
            bond_matrix[j, i] = 1
    for i, j in fixed_bonds:
        if i < n_atoms and j < n_atoms:
            bond_matrix[i, j] = 2
            bond_matrix[j, i] = 2

    return bond_matrix


def get_coords_minimised_v(coords, bonds, k, c, d0, tol, fixed_bonds, exponent=8):
    """Get the coordinates that minimise a FF with a bonds + repulsion FF where the repulsion is c/r^exponent"""
    # TODO divide and conquer?

    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)

    init_coords = coords.reshape(3 * n_atoms)
    bond_matrix = get_bond_matrix(n_atoms=len(coords), bonds=bonds, fixed_bonds=fixed_bonds)
    res = minimize(v, x0=init_coords, args=(bond_matrix, k, d0, c, exponent), method='CG', tol=tol, jac=dvdr)

    return res.x.reshape(n_atoms, 3)


def get_v(coords, bonds, k, c, d0, fixed_bonds, exponent=8):
    """Get the energy using a bond + repulsion FF where the repulsion is c/r^exponent"""
    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)

    init_coords = coords.reshape(3 * n_atoms)
    bond_matrix = get_bond_matrix(n_atoms=len(coords), bonds=bonds, fixed_bonds=fixed_bonds)

    return v(init_coords, bond_matrix, k, d0, c, exponent)


def get_atoms_rotated_stereocentres(species, atoms, rand):
    """If two stereocentres are bonded, rotate them randomly with respect to each other
    Arguments:
        species (autode.species.Species):
        atoms (list(autode.atoms.Atom)):
        rand (np.RandomState): random state
    """

    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    # Check on every pair of stereocenters
    for (i, j) in combinations(stereocentres, 2):
        if (i, j) not in species.graph.edges:
            continue

        # Don't rotate if the bond connecting the centers is a π-bond
        if species.graph.edges[i, j]['pi'] is True:
            logger.info('Stereocenters were π bonded – not rotating')
            continue

        try:
            left_idxs, right_idxs = split_mol_across_bond(species.graph, bond=(i, j))

        except CannotSplitAcrossBond:
            logger.warning('Splitting across this bond does not give two components - could have a ring')
            return atoms

        # Rotate the left hand side randomly
        rot_axis = atoms[i].coord - atoms[j].coord
        theta = 2*np.pi*rand.rand()
        idxs_to_rotate = left_idxs if i in left_idxs else right_idxs

        [atoms[n].rotate(axis=rot_axis, theta=theta, origin=atoms[i].coord) for n in idxs_to_rotate if n != i]

    return atoms


def add_dist_consts_for_stereocentres(species, dist_consts):
    """
    Add distances constraints across two bonded stereocentres, for example for a Z alkene, (hopefully) ensuring
    that in the conformer generation the stereochemistry is retained. Will also add distance constraints from
    one nearest neighbour to the other nearest neighbours for that chiral centre

    Arguments:
        species (autode.species.Species):
        dist_consts (dict): keyed with tuple of atom indexes and valued with the distance (Å), or None
    """
    if not are_coords_reasonable(coords=species.get_coordinates()):
        # TODO generate a reasonable initial structure: molassembler?
        logger.error('Cannot constrain stereochemistry if the initial structure is not sensible')
        return dist_consts

    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    # Get the stereocentres with 4 bonds as ~ chiral centres
    chiral_centres = [centre for centre in stereocentres if len(list(species.graph.neighbors(centre))) == 4]

    # Add distance constraints from one atom to the other 3 atoms to fix the configuration
    for chiral_centre in chiral_centres:
        neighbors = list(species.graph.neighbors(chiral_centre))
        atom_i = neighbors[0]

        for atom_j in neighbors[1:]:
            dist_consts[(atom_i, atom_j)] = species.get_distance(atom_i, atom_j)

    # Check on every pair of stereocenters
    for (atom_i, atom_j) in combinations(stereocentres, 2):

        # If they are not bonded don't alter
        if (atom_i, atom_j) not in species.graph.edges:
            continue

        # Add a single distance constraint between the nearest neighbours of each stereocentre
        for atom_i_neighbour in species.graph.neighbors(atom_i):
            for atom_j_neighbour in species.graph.neighbors(atom_j):
                if atom_i_neighbour != atom_j and atom_j_neighbour != atom_i:

                    # Fix the distance to the current value
                    dist_consts[(atom_i_neighbour, atom_j_neighbour)] = species.get_distance(atom_i_neighbour,
                                                                                             atom_j_neighbour)

    logger.info(f'Have {len(dist_consts)} distance constraint(s)')
    return dist_consts


def get_non_random_atoms(species):
    """Get the atoms that won't be randomised in the conformer generation. Stereocentres and nearest neighbours"""
    stereocentres = [node for node in species.graph.nodes if species.graph.nodes[node]['stereo'] is True]

    non_rand_atoms = deepcopy(stereocentres)
    for stereocentre in stereocentres:
        non_rand_atoms += list(species.graph.neighbors(stereocentre))

    if len(non_rand_atoms) > 0:
        logger.info(f'Not randomising atom index(es) {set(non_rand_atoms)}')

    return set(non_rand_atoms)


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


def get_coords_no_init_strucutre(atoms, species, d0, constrained_bonds):
    """
    Generate coordinates where no initial structure is present - this fixes(?) a problem for large molecule where
    if all the atoms are initially bonded and minimised then high energy minima are often found

    Args:
        atoms (list(autode.atoms.Atom)):
        species (autode.species.Species):
        d0 (np.ndarray):
        constrained_bonds (list):

    Returns:
        (np.ndarray) n_atoms x 3
    """
    # Minimise atoms with no bonds between them
    far_coords = get_coords_minimised_v(coords=np.array([atom.coord for atom in atoms]),
                                        bonds=species.graph.edges, fixed_bonds=constrained_bonds,
                                        k=0.0, c=0.1, d0=d0, tol=5E-3, exponent=2)
    coords = far_coords[:2]

    # Add the atoms one by one to the structure. Thanks to Dr. Cyrille Lavigne for this suggestion!
    for n in range(2, species.n_atoms):
        coords = get_coords_minimised_v(np.concatenate((coords, far_coords[len(coords):n+1])),
                                        bonds=species.graph.edges, fixed_bonds=constrained_bonds,
                                        k=0.1, c=0.1, d0=d0, tol=1E-3, exponent=2)

    # Perform a final minimisation
    coords = get_coords_minimised_v(coords=coords, bonds=species.graph.edges, fixed_bonds=constrained_bonds,
                                    k=1.0, c=0.01, d0=d0, tol=1E-5)
    return coords


def get_simanl_atoms(species, dist_consts=None, conf_n=0):
    """V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4
    Arguments:
        species (autode.species.Species): Species, Molecule, TSguess, TS
        dist_consts (dict): Key = tuple of atom indexes, Value = distance
        conf_n (int): Number of this conformer generated
    Returns:
        (np.ndarray): Coordinates of the generated conformer
    """
    xyz_filename = f'{species.name}_conf{conf_n}_siman.xyz'

    saved_atoms = get_atoms_from_generated_file(species, xyz_filename)
    if saved_atoms is not None:
        return saved_atoms

    # To generate the potential requires bonds between atoms defined in a
    # molecular graph
    if species.graph is None:
        raise NoMolecularGraph

    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = np.random.RandomState()
    atoms = get_atoms_rotated_stereocentres(species=species, atoms=deepcopy(species.atoms), rand=rand)

    # Add the distance constraints as fixed bonds
    d0 = get_ideal_bond_length_matrix(atoms=species.atoms, bonds=species.graph.edges())

    # Add distance constraints across stereocentres e.g. for a Z double bond then modify d0 appropriately
    dist_consts = add_dist_consts_for_stereocentres(species=species,
                                                    dist_consts={} if dist_consts is None else dist_consts)

    constrained_bonds = []
    for bond, length in dist_consts.items():
        i, j = bond
        d0[i, j] = length
        d0[j, i] = length
        constrained_bonds.append(bond)

    # Randomise coordinates that aren't fixed by shifting a maximum of autode.Config.max_atom_displacement in x, y, z
    fixed_atom_indexes = get_non_random_atoms(species=species)

    # Shift by a factor defined in the config file if the coordinates are reasonable but otherwise init in a 10 A cube
    initial_coords_are_reasonable = are_coords_reasonable(species.get_coordinates())

    if initial_coords_are_reasonable:
        factor = Config.max_atom_displacement / np.sqrt(3)
        [atom.translate(vec=factor * rand.uniform(-1, 1, 3)) for i, atom in enumerate(atoms) if i not in fixed_atom_indexes]
    else:
        # Randomise in a 10 Å cubic box
        [atom.translate(vec=rand.uniform(-5, 5, 3)) for atom in atoms]

    logger.info('Minimising species...')
    st = time()
    if initial_coords_are_reasonable:
        coords = get_coords_minimised_v(coords=np.array([atom.coord for atom in atoms]), bonds=species.graph.edges,
                                        k=1.0, c=0.01, d0=d0, tol=1E-5, fixed_bonds=constrained_bonds)

    else:
        coords = get_coords_no_init_strucutre(atoms, species, d0, constrained_bonds)

    logger.info(f'                 ... ({time()-st:.3f} s)')

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    # Print an xyz file so rerunning will read the file
    atoms_to_xyz_file(atoms=atoms, filename=xyz_filename)

    return atoms
