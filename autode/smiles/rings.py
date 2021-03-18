import numpy as np
from scipy.optimize import minimize


def minimise_ring_energy(atoms, dihedrals, close_idxs, r0):
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

    axes, rot_idxs, origins = [], [], []
    for dihedral in dihedrals:
        idx_i, idx_j = dihedral.mid_idxs

        origin = idx_i if idx_i in dihedral.rot_idxs else idx_j

        axes.append(dihedral.mid_idxs)
        rot_idxs.append(np.array(dihedral.rot_idxs, dtype=np.int))
        origins.append(origin)

    res = minimize(dihedral_rotations,
                   x0=init_dihedral_angles(dihedrals, atoms),
                   args=(coords, axes, rot_idxs, close_idxs, r0, origins),
                   method='CG',
                   tol=1E-2)

    # Apply the optimal set of dihedral rotations
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

    for angle, (i, j), idxs, origin in zip(angles, axes, rot_idxs, origins):

        axis = coords[i] - coords[j]
        axis /= np.linalg.norm(axis)

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


def init_dihedral_angles(dihedrals, atoms):
    """
    Precondition the optimiser on ring dihedrals

    Arguments:
        dihedrals (list(autode.smiles.builder.Dihedral)):
        atoms (list(autode.atoms.Atom)):

    Returns:
        (np.ndarray):
    """
    curr_angles = np.array([dihedral.value(atoms) for dihedral in dihedrals])

    # TODO: heuristics
    # return np.zeros(3)

    if len(dihedrals) == 3:
        ideal_angles = np.array([1.0472, -1.0472, 1.0472])  # ±60º

        return ideal_angles - curr_angles

    else:
        raise NotImplementedError

    return None
