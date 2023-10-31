import os
import numpy as np
import autode as ade
from copy import deepcopy
from typing import Dict, Optional, TYPE_CHECKING
from itertools import combinations
from scipy.optimize import minimize

from autode.conformers import Conformer
import autode.exceptions as ex
from autode.utils import log_time
from autode.input_output import xyz_file_to_atoms, atoms_to_xyz_file
from autode.mol_graphs import split_mol_across_bond
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.keywords import Keywords
    from autode.wrappers.methods import Method


def _get_bond_matrix(n_atoms, bonds, fixed_bonds):
    """
    Populate a bond matrix with 1 if i, j are bonded, 2 if i, j are bonded and
    fixed and 0 otherwise. Can support a partial structure with bonds to atoms
    that don't (yet) exist.

    ---------------------------------------------------------------------------
    Arguments:
        n_atoms (int):
        bonds (list(tuple)):
        fixed_bonds (list(tuple)):

    Returns:
        (np.ndarray): Bond matrix, shape = (n_atoms, n_atoms)
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


def _get_coords_energy(
    coords, bonds, k, c, d0, tol, fixed_bonds, exponent=8, fixed_idxs=None
):
    """
    Get the coordinates that minimise a FF with a bonds + repulsion FF
    where the repulsion is c/r^exponent

    ---------------------------------------------------------------------------
    Arguments:
        coords (np.ndarray): Initial coordinates, shape = (n_atoms, 3)
        bonds (list(tuple(int))): List of bonds
        fixed_bonds (list(tuple(int))): List of constrained bonds will use 10k
                    as the harmonic force constant
        k (float):
        c (float):

    Keyword Arguments:
        exponent (int): Exponent in the repulsive pairwise term

    Returns:
        (np.ndarray): Optimised coordinates, shape = (n_atoms, 3)
    """
    # TODO divide and conquer?
    from cconf_gen import v
    from cconf_gen import dvdr

    n_atoms = len(coords)
    os.environ["OMP_NUM_THREADS"] = str(1)

    bond_matrix = _get_bond_matrix(
        n_atoms=len(coords), bonds=bonds, fixed_bonds=fixed_bonds
    )

    if fixed_idxs is None:
        fixed_idxs = np.array([], dtype=int)

    res = minimize(
        v,
        x0=coords.reshape(3 * n_atoms),
        args=(bond_matrix, k, d0, c, exponent, fixed_idxs),
        method="CG",
        tol=tol,
        jac=dvdr,
    )

    return res.x.reshape(n_atoms, 3), res.fun


def _get_v(coords, bonds, k, c, d0, fixed_bonds, exponent=8):
    """Get the energy using a bond + repulsion FF where

    V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^exponent

    ---------------------------------------------------------------------------
    Arguments:
        coords (np.ndarray): shape = (n_atoms, 3)
        bonds (list(tuple(int))): List of bonds
        fixed_bonds (list(tuple(int))): List of constrained bonds will use 10k
                    as the harmonic force constant
        k (float):
        c (float):
        exponent (int): Exponent in the repulsive pairwise term

    Returns:
        (float): Energy
    """
    from cconf_gen import v

    n_atoms = len(coords)
    os.environ["OMP_NUM_THREADS"] = str(1)

    init_coords = coords.reshape(3 * n_atoms)
    bond_matrix = _get_bond_matrix(
        n_atoms=n_atoms, bonds=bonds, fixed_bonds=fixed_bonds
    )

    return v(init_coords, bond_matrix, k, d0, c, exponent)


def _get_atoms_rotated_stereocentres(species, atoms, rand):
    """If two stereocentres are bonded, rotate them randomly with respect
    to each other

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        atoms (list(autode.atoms.Atom)):
        rand (np.RandomState): random state

    Returns:
        (list(autode.atoms.Atom)): Atoms
    """

    stereocentres = [
        node
        for node in species.graph.nodes
        if species.graph.nodes[node]["stereo"] is True
    ]

    # Check on every pair of stereocenters
    for i, j in combinations(stereocentres, 2):
        if (i, j) not in species.graph.edges:
            continue

        # Don't rotate if the bond connecting the centers is a π-bond
        if species.graph.edges[i, j]["pi"] is True:
            logger.info("Stereocenters were π bonded – not rotating")
            continue

        try:
            left_idxs, right_idxs = split_mol_across_bond(
                species.graph, bond=(i, j)
            )

        except ex.CannotSplitAcrossBond:
            logger.warning(
                "Splitting across this bond does not give two "
                "components - could have a ring"
            )
            return atoms

        # Rotate the left hand side randomly
        rot_axis = atoms[i].coord - atoms[j].coord
        theta = 2 * np.pi * rand.rand()
        idxs_to_rotate = left_idxs if i in left_idxs else right_idxs

        # Rotate all the atoms to the left of this bond, missing out i as that
        # is the origin for rotation and thus won't move
        for n in idxs_to_rotate:
            if n == i:
                continue
            atoms[n].rotate(axis=rot_axis, theta=theta, origin=atoms[i].coord)

    return atoms


def _add_dist_consts_for_stereocentres(species, dist_consts):
    """
    Add distances constraints across two bonded stereocentres, for example
    for a Z alkene, (hopefully) ensuring that in the conformer generation the
    stereochemistry is retained. Will also add distance constraints from
    one nearest neighbour to the other nearest neighbours for that chiral
    centre

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        dist_consts (dict): keyed with tuple of atom indexes and valued with
                    the distance (Å), or None

    Returns:
        (dict): Distance constraints
    """
    if not ade.geom.are_coords_reasonable(coords=species.coordinates):
        # TODO generate a reasonable initial structure: molassembler?
        logger.error(
            "Cannot constrain stereochemistry if the initial "
            "structure is not sensible"
        )
        return dist_consts

    stereocentres = [
        node
        for node in species.graph.nodes
        if species.graph.nodes[node]["stereo"] is True
    ]

    # Get the stereocentres with 4 bonds as ~ chiral centres
    chiral_centres = [
        centre
        for centre in stereocentres
        if len(list(species.graph.neighbors(centre))) == 4
    ]

    # Add distance constraints from one atom to the other 3 atoms to fix the
    # configuration
    for chiral_centre in chiral_centres:
        neighbors = list(species.graph.neighbors(chiral_centre))
        atom_i = neighbors[0]

        for atom_j in neighbors[1:]:
            dist_consts[(atom_i, atom_j)] = species.distance(atom_i, atom_j)

    # Check on every pair of stereocenters
    for atom_i, atom_j in combinations(stereocentres, 2):
        # If they are not bonded don't alter
        if (atom_i, atom_j) not in species.graph.edges:
            continue

        # Add a single distance constraint between the nearest neighbours of
        # each stereocentre
        for i_neighbour in species.graph.neighbors(atom_i):
            for j_neighbour in species.graph.neighbors(atom_j):
                if i_neighbour != atom_j and j_neighbour != atom_i:
                    # Fix the distance to the current value
                    dist = species.distance(i_neighbour, j_neighbour)
                    dist_consts[(i_neighbour, j_neighbour)] = dist

    logger.info(f"Have {len(dist_consts)} distance constraint(s)")
    return dist_consts


def _get_non_random_atoms(species):
    """
    Get the atoms that won't be randomised in the conformer generation.
    Stereocentres and nearest neighbours

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

    Returns:
        (set(int)): Atoms indexes to not randomise
    """
    stereocentres = [
        node
        for node in species.graph.nodes
        if species.graph.nodes[node]["stereo"] is True
    ]

    non_rand_atoms = deepcopy(stereocentres)
    for stereocentre in stereocentres:
        non_rand_atoms += list(species.graph.neighbors(stereocentre))

    if len(non_rand_atoms) > 0:
        logger.info(f"Not randomising atom index(es) {set(non_rand_atoms)}")

    return np.array(list(set(non_rand_atoms)), dtype=int)


def _get_atoms_from_generated_file(species, xyz_filename):
    """
    Get atoms from a previously generated  .xyz file, if the atoms match

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        xyz_filename (str):

    Returns:
        (list(autode.atoms.Atoms)) or None: Atoms from file
    """

    if not os.path.exists(xyz_filename):
        return None

    atoms = xyz_file_to_atoms(filename=xyz_filename)

    if len(atoms) != species.n_atoms:
        return None

    all_atoms_match = all(
        atoms[i].label == species.atoms[i].label
        for i in range(species.n_atoms)
    )

    if all_atoms_match:
        logger.info("Conformer has already been generated")
        return atoms

    return None


def _get_coords_no_init_structure(atoms, species, d0, constrained_bonds):
    """
    Generate coordinates where no initial structure is present - this fixes(?)
     a problem for large molecule where if all the atoms are initially bonded
     and minimised then high energy minima are often found

    Args:
        atoms (list(autode.atoms.Atom)):
        species (autode.species.Species):
        d0 (np.ndarray):
        constrained_bonds (list):

    Returns:
        (np.ndarray): Optimised coordinates, shape = (n_atoms, 3)
    """
    # Minimise atoms with no bonds between them
    far_coords, _ = _get_coords_energy(
        coords=np.array([atom.coord for atom in atoms]),
        bonds=species.graph.edges,
        fixed_bonds=constrained_bonds,
        k=0.0,
        c=0.1,
        d0=d0,
        tol=5e-3,
        exponent=2,
    )
    coords = far_coords[:2]

    # Add the atoms one by one to the structure. Thanks to Dr. Cyrille Lavigne
    #  for this suggestion!
    for n in range(2, species.n_atoms):
        new_coords = np.concatenate((coords, far_coords[len(coords) : n + 1]))
        coords, _ = _get_coords_energy(
            new_coords,
            bonds=species.graph.edges,
            fixed_bonds=constrained_bonds,
            k=0.1,
            c=0.1,
            d0=d0,
            tol=1e-3,
            exponent=2,
        )

    # Perform a final minimisation
    coords, energy = _get_coords_energy(
        coords=coords,
        bonds=species.graph.edges,
        fixed_bonds=constrained_bonds,
        k=1.0,
        c=0.01,
        d0=d0,
        tol=1e-5,
    )
    return coords, energy


@log_time(prefix="Generated RR atoms in:", units="s")
def get_simanl_atoms(
    species: "Species",
    dist_consts: Optional[Dict] = None,
    conf_n: int = 0,
    save_xyz: bool = True,
    also_return_energy: bool = False,
):
    r"""
    Use a bonded + repulsive force field to generate 3D structure for a
    species. If the initial coordinates are reasonable e.g. from a previously
    generated 3D structure then add random displacement vectors and minimise
    to generate a conformer. Otherwise add atoms to the box sequentially
    until all atoms have been added, which generates a qualitatively reasonable
    3D geometry which should be optimised using a electronic structure method::

        V(x) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^n

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

        dist_consts (dict): Key = tuple of atom indexes, Value = distance

        conf_n (int): Number of this conformer

        save_xyz (bool): Whether or not to save a .xyz file of the structure
                         for fast reloading

        also_return_energy (bool): Whether or not to return the energy

    Returns:
        (list(autode.atoms.Atom)): Atoms
    """
    xyz_filename = f"{species.name}_conf{conf_n}_siman.xyz"

    saved_atoms = _get_atoms_from_generated_file(species, xyz_filename)
    if saved_atoms is not None and not also_return_energy:
        return saved_atoms

    # To generate the potential requires bonds between atoms defined in a
    # molecular graph
    if species.graph is None:
        raise ex.NoMolecularGraph

    # Initialise a new random seed and make a copy of the species' atoms.
    # RandomState is thread safe
    rand = np.random.RandomState()
    atoms = _get_atoms_rotated_stereocentres(
        species=species, atoms=deepcopy(species.atoms), rand=rand
    )

    # Add the distance constraints as fixed bonds
    d0 = species.graph.eqm_bond_distance_matrix

    # Add distance constraints across stereocentres e.g. for a Z double bond
    # then modify d0 appropriately
    curr_dist_consts = {} if dist_consts is None else dist_consts
    distance_constraints = _add_dist_consts_for_stereocentres(
        species=species, dist_consts=curr_dist_consts
    )

    constrained_bonds = []
    for bond, length in distance_constraints.items():
        i, j = bond
        d0[i, j] = length
        d0[j, i] = length
        constrained_bonds.append(bond)

    # Randomise coordinates that aren't fixed by shifting a maximum of
    # autode.Config.max_atom_displacement in x, y, z
    fixed_atom_indexes = _get_non_random_atoms(species=species)

    # Shift by a factor defined in the config file if the coordinates are
    # reasonable but otherwise init in a 10 A cube
    reasonable_init_coords = ade.geom.are_coords_reasonable(
        species.coordinates
    )

    if reasonable_init_coords:
        factor = ade.Config.max_atom_displacement / np.sqrt(3)
        for i, atom in enumerate(atoms):
            if i not in fixed_atom_indexes:
                atom.translate(vec=factor * rand.uniform(-1, 1, 3))
    else:
        # Randomise in a 10 Å cubic box
        [atom.translate(vec=rand.uniform(-5, 5, 3)) for atom in atoms]

    if reasonable_init_coords:
        init_coords = np.array([atom.coord for atom in atoms])
        coords, energy = _get_coords_energy(
            coords=init_coords,
            bonds=species.graph.edges,
            k=1.0,
            c=0.01,
            d0=d0,
            tol=1e-5,
            fixed_idxs=fixed_atom_indexes,
            fixed_bonds=constrained_bonds,
        )
    else:
        coords, energy = _get_coords_no_init_structure(
            atoms, species, d0, constrained_bonds
        )

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    # Print an xyz file so rerunning will read the file
    if save_xyz:
        atoms_to_xyz_file(atoms=atoms, filename=xyz_filename)

    if also_return_energy:
        logger.info(f"E_RR = {energy:.6f}")
        return atoms, energy

    return atoms


def get_simanl_conformer(
    species: "Species",
    dist_consts: Optional[Dict] = None,
    conf_n: int = 0,
    save_xyz: bool = True,
) -> "Conformer":
    """
    Generate a conformer of a species using randomise+relax with a simple FF
    (see get_simanl_atoms). Example

    .. code-block:: Python
        >>> import autode as ade
        >>> from autode.conformers.conf_gen import get_simanl_conformer
        >>> mol = ade.Molecule(smiles='CCCC', name='butane')
        >>> conf0 = get_simanl_conformer(mol, conf_n=0, save_xyz=False)
        Conformer(butane_conf0, n_atoms=14, charge=0, mult=1)

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

        dist_consts (dict): Key = tuple of atom indexes, Value = distance

        conf_n (int): Number of this conformer

        save_xyz (bool): Whether or not to save a .xyz file of the structure

    Returns:
        (autode.conformers.Conformer): Conformer
    """

    conformer = Conformer(
        species=species,
        name=f"{species.name}_conf{conf_n}",
        dist_consts=dist_consts,
    )

    atoms, energy = get_simanl_atoms(
        species,
        dist_consts=dist_consts,
        conf_n=conf_n,
        save_xyz=save_xyz,
        also_return_energy=True,
    )
    conformer.atoms = atoms
    conformer.energy = energy

    return conformer
