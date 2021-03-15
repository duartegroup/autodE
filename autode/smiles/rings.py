import numpy as np
from scipy.optimize import minimize


def minimise_ring_energy(atoms, pairs_rot_idxs, close_idxs, r0):
    """
    For a set of atoms minimise the energy with respect to dihedral rotation,
    where the energy is...


    atoms will be modified in place with respect to the rotation

    Arguments:
        atoms:
        pairs_rot_idxs:
        close_idxs:
        r0 (float):
    """
    coords = np.array([a.coord for a in atoms], copy=True)
    axes = [coords[idx_i] - coords[idx_j]
            for (idx_i, idx_j) in pairs_rot_idxs.keys()]

    # normalise
    axes = [axis / np.linalg.norm(axis) for axis in axes]

    rot_idxs = [np.array(idxs, dtype=np.int)
                for idxs in pairs_rot_idxs.values()]

    res = minimize(dihedral_rotations,
                   x0=np.zeros(len(pairs_rot_idxs)),
                   args=(coords, axes, rot_idxs, close_idxs, r0),
                   method='CG',
                   tol=1E-2)
    print(res)

    # apply the optimal set of dihedral rotations
    new_coords = dihedral_rotations(res.x, coords, axes, rot_idxs, close_idxs,
                                    r0, return_energy=False)

    # and set the new coordinates
    for i, atom in enumerate(atoms):
        atom.coord = new_coords[i]

    return None


def dihedral_rotations(angles, coords, axes, rot_idxs, close_idxs, r0,
                       return_energy=True):
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

    for angle, axis, idxs in zip(angles, axes, rot_idxs):

        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2*(bd - ac)],
                               [2*(bc - ad), aa + cc - bb - dd, 2*(cd + ab)],
                               [2*(bd + ac), 2*(cd - ab), aa + dd - bb - cc]])

        coords[idxs] = np.dot(rot_matrix, coords[idxs].T).T

    if return_energy:
        idx_i, idx_j = close_idxs
        r = np.linalg.norm(coords[idx_i] - coords[idx_j])
        return (r - r0)**2

    return coords
