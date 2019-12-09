import numpy as np


def xyz2coord(xyzs):
    """
    For a set of xyzs in the form e.g [[C, 0.0, 0.0, 0.0], ...] convert to a np array of coordinates, containing just
    just the x, y, z coordinates

    Arguments:
        xyzs {list(list)} -- List of xyzs

    Returns:
        {np.array} -- array of coords
    """
    if isinstance(xyzs[0], list):
        return np.array([np.array(line[1:4]) for line in xyzs])
    else:
        return np.array(xyzs[1:4])


def coords2xyzs(coords, old_xyzs):
    """Insert a set of coordinates into a set of xyzs to a new list of xyzs

    Arguments:
        coords {np.array} -- array of coordinates
        old_xyzs {list(lists)} -- list of old xyzs

    Returns:
        list(list) -- coords in xyz form
    """
    if isinstance(old_xyzs[0], list):
        assert len(old_xyzs) == len(coords)
        return [[old_xyzs[n][0]] + coords[n].tolist() for n in range(len(old_xyzs))]
    else:
        assert len(old_xyzs) == 4
        assert len(coords) == 3
        return [old_xyzs[0]] + coords.tolist()


def get_shifted_xyzs_linear_interp(xyzs, bonds, final_distances):
    """For a geometry defined by a set of xyzs, set the constrained bonds to the correct lengths

    Arguments:
        xyzs {list(list)} -- list of xyzs
        bonds {list(tuple)} -- List of bond ids on for which the final_distances apply
        final_distances {list(float)} -- List of final bond distances for the bonds

    Returns:
        {list(list)} -- shifted xyzs
    """

    coords = xyz2coord(xyzs)
    atoms_and_shift_vecs = {}

    for n, bond in enumerate(bonds):
        atom_a, atom_b = bond
        ab_vec = coords[atom_b] - coords[atom_a]
        ab_dist = np.linalg.norm(ab_vec)
        ab_final_dist = final_distances[n]

        ab_norm_vec = ab_vec / ab_dist

        atoms_and_shift_vecs[atom_b] = 0.5 * \
            (ab_final_dist - ab_dist) * ab_norm_vec
        atoms_and_shift_vecs[atom_a] = -0.5 * \
            (ab_final_dist - ab_dist) * ab_norm_vec

    for n, coord in enumerate(coords):
        if n in atoms_and_shift_vecs.keys():
            coord += atoms_and_shift_vecs[n]

    new_xyzs = coords2xyzs(coords, old_xyzs=xyzs)

    return new_xyzs


def calc_distance_matrix(xyzs):
    """Calculate a distance matrix

    Arguments:
        xyzs {list(list)} -- list of xyzs

    Returns:
        np.array -- array of distances between all the atoms
    """

    n_atoms = len(xyzs)
    coords = xyz2coord(xyzs)
    distance_matrix = np.zeros([n_atoms, n_atoms])

    for atom_i in range(n_atoms):
        for atom_j in range(n_atoms):
            dist = np.linalg.norm(coords[atom_i] - coords[atom_j])
            distance_matrix[atom_i, atom_j] = dist

    return distance_matrix


def get_neighbour_list(atom_i, mol):
    """Calculate a neighbour list from atom i as a list of atom labels

    Arguments:
        atom_i {int} -- index of the atom
        mol {molecule object} -- Molecule object

    Returns:
        list -- list of atom ids in ascending distance away from atom_i
    """
    distance_vector = mol.distance_matrix[atom_i]
    dists_and_atom_labels = {}
    for atom_j, dist in enumerate(distance_vector):
        dists_and_atom_labels[dist] = mol.xyzs[atom_j][0]

    atom_label_neighbour_list = []
    for dist, atom_label in sorted(dists_and_atom_labels.items()):
        atom_label_neighbour_list.append(atom_label)

    return atom_label_neighbour_list


def calc_rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Arguments:
        axis {np.array} -- axis to be rotated about
        theta {float} -- angle in radians to be rotated

    Returns:
        {np.array} -- rotation matrix
    """
    axis = np.asarray(axis)
    axis = axis/np.linalg.norm(axis)
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


i = np.array([1.0, 0.0, 0.0])
j = np.array([0.0, 1.0, 0.0])
k = np.array([0.0, 0.0, 1.0])
