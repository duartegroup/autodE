import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from autode.log import logger


def are_coords_reasonable(coords):
    """
    Determine if a set of coords are reasonable. No distances can be < 0.7 Å
    and if there are more than 4 atoms ensure they do not all lie in the same
    plane. The latter possibility arises from RDKit's conformer generation
    algorithm breaking

    Arguments:
        coords (np.ndarray): Species coordinates as a n_atoms x 3 array

    Returns:
        bool:
    """
    n_atoms = len(coords)

    # Generate a n_atoms x n_atoms matrix with ones on the diagonal
    dist_mat = distance_matrix(coords, coords) + np.identity(n_atoms)

    if np.min(dist_mat) < 0.7:
        logger.warning('There is a distance < 0.7 Å. Structure is *not* '
                       'sensible')
        return False

    if n_atoms > 4:
        if all([coord[2] == 0.0 for coord in coords]):
            logger.warning('RDKit likely generated a wrong geometry. Structure'
                           ' is *not* sensible')
            return False

    return True


def proj(u, v):
    """
    Calculate the projection of v onto the direction of u. Useful for
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    Arguments:
        u (np.ndarray):

        v (np.ndarray):

    Returns:
        (np.ndarray): proj_u(v)
    """

    u_hat = u / np.linalg.norm(u)
    return np.dot(u, v) * u_hat


def get_atoms_linear_interp(atoms, bonds, final_distances):
    """For a geometry defined by a set of xyzs, set the constrained bonds to
    the correct lengths

    Arguments:
        atoms (list(autode.atoms.Atom)): list of atoms
        bonds (list(tuple)): List of bond ids on for which the final_distances
                             apply
        final_distances (list(float)): List of final bond distances for the
                                       bonds

    Returns:
        (list(autode.atoms.Atom)): Shifted atoms
    """

    coords = np.array([atom.coord for atom in atoms])
    atoms_and_shift_vecs = {}

    for n, bond in enumerate(bonds):
        atom_a, atom_b = bond
        ab_vec = coords[atom_b] - coords[atom_a]
        d_crr = np.linalg.norm(ab_vec)
        d_final = final_distances[n]

        ab_norm_vec = ab_vec / d_crr

        atoms_and_shift_vecs[atom_b] = 0.5 * (d_final - d_crr) * ab_norm_vec
        atoms_and_shift_vecs[atom_a] = -0.5 * (d_final - d_crr) * ab_norm_vec

    for n, coord in enumerate(coords):
        if n in atoms_and_shift_vecs.keys():
            coord += atoms_and_shift_vecs[n]

        atoms[n].coord = coord

    return atoms


def get_rot_mat_kabsch(p_matrix, q_matrix):
    """
    Get the optimal rotation matrix with the Kabsch algorithm. Notation is from
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    Arguments:
        p_matrix (np.ndarray):
        q_matrix (np.ndarray):

    Returns:
        (np.ndarray): rotation matrix
    """

    h = np.matmul(p_matrix.transpose(), q_matrix)
    u, _, vh = np.linalg.svd(h)
    d = np.linalg.det(np.matmul(vh.transpose(), u.transpose()))
    int_mat = np.identity(3)
    int_mat[2, 2] = d
    rot_matrix = np.matmul(np.matmul(vh.transpose(), int_mat), u.transpose())

    return rot_matrix


def get_rot_mat_euler_from_terms(a, b, c, d):
    """3D rotation matrix from terms unique terms in the matrix"""

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                           [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                           [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return rot_matrix


def get_rot_mat_euler(axis, theta):
    """
    Compute the 3D rotation matrix using the Euler Rodrigues formula
    https://en.wikipedia.org/wiki/Euler–Rodrigues_formula
    for an anticlockwise rotation of theta radians about a given axis

    Arguments:
        axis (np.ndarray): Axis to rotate in. shape = (3,)
        theta (float): Angle in radians (float)

    Returns:
        (np.ndarray): Rotation matrix. shape = (3, 3)
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)   # Normalise

    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    rot_matrix = get_rot_mat_euler_from_terms(a=a, b=b, c=c, d=d)

    return rot_matrix


def get_neighbour_list(species, atom_i, index_set):
    """Calculate a neighbour list from atom i as a list of atom labels

    Arguments:
        atom_i (int): index of the atom
        species (autode.species.Species):
        index_set (set(int) | None): Indexes that are possible neighbours for
                                     atom_i, if None then all atoms are ok

    Returns:
        (list(int)): list of atom ids in ascending distance away from i
    """
    coords = species.coordinates
    distance_vector = cdist(np.array([coords[atom_i]]), coords)[0]

    dists_and_atom_labels = {}
    for atom_j, dist in enumerate(distance_vector):

        if index_set is not None and atom_j not in index_set:
            continue

        dists_and_atom_labels[dist] = species.atoms[atom_j].label

    atom_label_neighbour_list = []
    for dist, atom_label in sorted(dists_and_atom_labels.items()):
        atom_label_neighbour_list.append(atom_label)

    return atom_label_neighbour_list


def get_distance_constraints(species):
    """Set all the distance constraints required in an optimisation as the
    active bonds

    Arguments:
        species (autode.species.Species):

    Returns:
        (dict): Keyed with atom indexes for the active atoms (tuple) and
                equal to the constrained value
    """
    distance_constraints = {}

    if species.graph is None:
        logger.warning('Molecular graph was not set cannot find any distance '
                       'constraints')
        return None

    # Add the active edges(/bonds) in the molecular graph to the dict, value
    # being the current distance
    for edge in species.graph.edges:

        if species.graph.edges[edge]['active']:
            distance_constraints[edge] = species.distance(*edge)

    return distance_constraints


def calc_heavy_atom_rmsd(atoms1, atoms2):
    """
    Calculate the RMSD between two sets of atoms considering only the 'heavy'
    atoms, i.e. the non-hydrogen atoms

    Arguments:
        atoms1 (list(autode.atoms.Atom)):
        atoms2 (list(autode.atoms.Atom)):

    Returns:
        (float): RMSD between the two sets
    """
    if len(atoms1) != len(atoms2):
        raise ValueError('RMSD must be computed between atom lists of the'
                         f'same length: {len(atoms1)} =/= {len(atoms2)}')

    coords1 = np.array([atom.coord for atom in atoms1 if atom.label != 'H'])
    coords2 = np.array([atom.coord for atom in atoms2 if atom.label != 'H'])

    if len(coords1) == 0 or len(coords2) == 0:
        logger.warning('No heavy atoms! assuming a zero RMSD')
        return 0.0

    return calc_rmsd(coords1, coords2)


def calc_rmsd(coords1, coords2):
    """Calculate the RMSD between two sets of coordinates using the Kabsch
    algorithm

    Arguments:
        coords1 (np.ndarray): shape = (n, 3)
        coords2 (np.ndarray): shape = (n ,3)

    Returns:
        (float): Root mean squared distance
    """
    assert coords1.shape == coords2.shape

    p_mat = np.array(coords2, copy=True)
    p_mat -= np.average(p_mat, axis=0)

    q_mat = np.array(coords1, copy=True)
    q_mat -= np.average(q_mat, axis=0)

    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)

    fitted_coords = np.dot(rot_mat, p_mat.T).T
    return np.sqrt(np.average(np.square(fitted_coords - q_mat)))


def get_points_on_sphere(n_points, r=1):
    """
    Find n evenly spaced points on a sphere using the "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno, 2004.

    Arguments:
        n_points (int): number of points to generate
        r (float): radius of the sphere

    Returns:
        (list(np.ndarray))
    """
    points = []

    a = 4.0 * np.pi * r**2 / n_points
    d = np.sqrt(a)
    m_theta = int(np.round(np.pi / d))
    d_theta = np.pi / m_theta
    d_phi = a / d_theta

    for m in range(m_theta):
        theta = np.pi * (m + 0.5) / m_theta
        m_phi = int(np.round(2.0 * np.pi * np.sin(theta)/d_phi))

        for n in range(m_phi):
            phi = 2.0 * np.pi * n / m_phi
            point = np.array([r * np.sin(theta) * np.cos(phi),
                              r * np.sin(theta) * np.sin(phi),
                              r * np.cos(theta)])

            points.append(point)

    return points


def symm_matrix_from_ltril(array):
    """
    Construct a symmetric matrix from the lower triangular elements e.g.::

        array = [0, 1, 2] ->  array([[0, 1],
                                     [1, 2]])

    Arguments:
        array (list(float) | np.array):

    Returns:
        (np.ndarray):
    """
    n = int((np.sqrt(8*len(array) + 1) - 1)/2)

    matrix = np.zeros(shape=(n, n), dtype='f8')

    try:
        matrix[np.tril_indices(n=n, k=0)] = np.array(array)

    except ValueError:
        raise ValueError('Array was not the correct shape to be broadcast '
                         'into the lower triangle. Need N(N+1)/2 elements'
                         'for an NxN array')

    # Symmetrise by making the upper triangular elements to the lower
    lower_idxs = np.tril_indices(n=n, k=-1)
    matrix.T[lower_idxs] = matrix[lower_idxs]

    return matrix
