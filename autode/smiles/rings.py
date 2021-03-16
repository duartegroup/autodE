import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix


def minimise_ring_energy(atoms, pairs_rot_idxs, close_idxs, r0):
    """
    For a set of atoms minimise the energy with respect to dihedral rotation,
    where the energy is...


    atoms will be modified in place with respect to the rotation

    --------------------------------------------------------------------------
    Arguments:
        atoms (list(autode.smiles.SMILESAtom)): Atoms, a subset of which need
                                                rotation

        pairs_rot_idxs (dict): Dictionary keyed with central two atoms indexes
                               of a dihedral and the values as all the atoms
                               to rotate

        close_idxs (tuple(int)): Atom indexes for the bond to close

        r0 (float): Ideal distance (Å) for the closed bond
    """
    coords = np.array([a.coord for a in atoms], copy=True)

    # normalised rotation axes
    axes = [coords[idx_i] - coords[idx_j]
            for (idx_i, idx_j) in pairs_rot_idxs.keys()]

    axes = [axis / np.linalg.norm(axis) for axis in axes]

    # atom indexes to rotate
    rot_idxs = [np.array(idxs, dtype=np.int)
                for idxs in pairs_rot_idxs.values()]

    # origins for the rotation to be applied
    origins = [idx_i if idx_i in idxs else idx_j
               for (idx_i, idx_j), idxs in pairs_rot_idxs.items()]

    res = minimize(dihedral_rotations,
                   x0=np.zeros(len(pairs_rot_idxs)),
                   args=(coords, axes, rot_idxs, close_idxs, r0, origins),
                   method='CG',
                   tol=1E-2)
    print(res)

    # apply the optimal set of dihedral rotations
    new_coords = dihedral_rotations(res.x, coords, axes, rot_idxs, close_idxs,
                                    r0, origins,
                                    return_energy=False)

    # and set the new coordinates
    for i, atom in enumerate(atoms):
        atom.coord = new_coords[i]

    return None


def dihedral_rotations(angles, coords, axes, rot_idxs, close_idxs, r0,
                       origins, return_energy=True):
    """
    Perform a set of dihedral rotations and calculate the 'energy'

    Arguments:
        angles:
        coords:
        axes:
        rot_idxs:
        close_idxs:
        r0:

    Keyword Arguments:
        return_energy (bool):

    Returns:
        (float | np.ndarray):
    """
    coords = np.copy(coords)

    for angle, axis, idxs, origin in zip(angles, axes, rot_idxs, origins):

        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2*(bd - ac)],
                               [2*(bc - ad), aa + cc - bb - dd, 2*(cd + ab)],
                               [2*(bd + ac), 2*(cd - ab), aa + dd - bb - cc]])

        origin_coord = np.copy(coords[origin])
        coords -= origin_coord
        coords[idxs] = np.dot(rot_matrix, coords[idxs].T).T
        coords += origin_coord

    if return_energy:
        idx_i, idx_j = close_idxs
        r = np.linalg.norm(coords[idx_i] - coords[idx_j])
        return (r - r0)**2

    return coords
