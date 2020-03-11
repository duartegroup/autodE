import numpy as np
from scipy.spatial import distance_matrix
from autode.log import logger


def are_coords_reasonable(coords):
    """
    Determine if a set of coords are reasonable. No distances can be < 0.7 Å and if there are more than 4 atoms ensure
    they do not all lie in the same plane. The latter possibility arises from RDKit's conformer generation algorithm
    breaking


    Arguments:
        coords (np.ndarray): Species coordinates as a n_atoms x 3 array

    Returns:
        bool:
    """

    n_atoms = len(coords)

    # Generate a n_atoms x n_atoms matrix
    distance_matrix_unity_diag = distance_matrix(coords, coords) + np.identity(n_atoms)

    if np.min(distance_matrix_unity_diag) < 0.7:
        logger.warning('There is a distance < 0.7 Å. Structure is *not* sensible')
        return False

    if n_atoms > 4:
        if all([coord[2] == 0.0 for coord in coords]):
            logger.warning('RDKit likely generated a wrong geometry. Structure is *not* sensible')
            return False

    return True


def get_shifted_xyzs_linear_interp(xyzs, bonds, final_distances):
    """For a geometry defined by a set of xyzs, set the constrained bonds to the correct lengths

    Arguments:
        xyzs (list(list)): list of xyzs
        bonds (list(tuple)): List of bond ids on for which the final_distances apply
        final_distances (list(float)): List of final bond distances for the bonds

    Returns:
        list(list): shifted xyzs
    """

    coords = xyz2coord(xyzs)
    atoms_and_shift_vecs = {}

    for n, bond in enumerate(bonds):
        atom_a, atom_b = bond
        ab_vec = coords[atom_b] - coords[atom_a]
        ab_dist = np.linalg.norm(ab_vec)
        ab_final_dist = final_distances[n]

        ab_norm_vec = ab_vec / ab_dist

        atoms_and_shift_vecs[atom_b] = 0.5 * (ab_final_dist - ab_dist) * ab_norm_vec
        atoms_and_shift_vecs[atom_a] = -0.5 * (ab_final_dist - ab_dist) * ab_norm_vec

    for n, coord in enumerate(coords):
        if n in atoms_and_shift_vecs.keys():
            coord += atoms_and_shift_vecs[n]

    new_xyzs = coords2xyzs(coords, old_xyzs=xyzs)

    return new_xyzs


def get_neighbour_list(atom_i, mol):
    """Calculate a neighbour list from atom i as a list of atom labels

    Arguments:
        atom_i (int): index of the atom
        mol (molecule object): Molecule object

    Returns:
        list: list of atom ids in ascending distance away from atom_i
    """
    distance_vector = mol.distance_matrix[atom_i]
    dists_and_atom_labels = {}
    for atom_j, dist in enumerate(distance_vector):
        dists_and_atom_labels[dist] = mol.xyzs[atom_j][0]

    atom_label_neighbour_list = []
    for dist, atom_label in sorted(dists_and_atom_labels.items()):
        atom_label_neighbour_list.append(atom_label)

    return atom_label_neighbour_list




i = np.array([1.0, 0.0, 0.0])
j = np.array([0.0, 1.0, 0.0])
k = np.array([0.0, 0.0, 1.0])
