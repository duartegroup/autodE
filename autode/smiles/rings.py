import numpy as np
from time import time
from scipy.optimize import minimize
from autode.log import logger
from autode.geom import get_rot_mat_euler


def minimise_ring_energy(atoms, dihedrals, close_idxs, r0):
    """
    For a set of atoms minimise the energy with respect to dihedral rotation,
    where the energy is...


    atoms will be modified in place with respect to the rotation

    --------------------------------------------------------------------------
    Arguments:
        atoms (list(autode.smiles.SMILESAtom)): Atoms, a subset of which need
                                                rotation

        dihedrals (list(autode.smiles.builder.Dihedral)): Dihedrals to rotate
                                                          about

        close_idxs (tuple(int)): Atom indexes for the bond to close

        r0 (float): Ideal distance (Å) for the closed bond
    """
    start_time = time()

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
    # print(res)

    # Apply the optimal set of dihedral rotations
    new_coords = dihedral_rotations(res.x, coords, axes, rot_idxs, close_idxs,
                                    r0, origins,
                                    return_energy=False)

    # and set the new coordinates
    for i, atom in enumerate(atoms):
        atom.coord = new_coords[i]

    logger.info(f'Closed ring in {(time() - start_time)*1000:.2f} ms')
    return None


def dihedral_rotations(angles, coords, axes, rot_idxs, close_idxs, r0,
                       origins, return_energy=True):
    """
    Perform a set of dihedral rotations and calculate the 'energy', defined
    by a harmonic in the difference between the ideal X-Y distance (r0) and
    it's current value, where X, Y are the two atoms in the tuple close_idxs

    Arguments:
        angles (np.ndarray):  shape = (n,)
        coords (np.ndarray):  shape = (n, 3)
        axes (list(tuple(int))):
        rot_idxs (list(list(int))):
        close_idxs (tuple(int)):
        origins (list(int)):
        r0 (float):

    Keyword Arguments:
        return_energy (bool):

    Returns:
        (float | np.ndarray):
    """
    coords = np.copy(coords)

    for angle, (i, j), idxs, origin in zip(angles, axes, rot_idxs, origins):

        rot_matrix = get_rot_mat_euler(axis=coords[i] - coords[j],
                                       theta=angle)

        origin_coord = np.copy(coords[origin])

        # Shift, rotate and shift back
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
    n_angles = len(dihedrals)
    logger.info(f'Calculating ideal ∆ϕ for {n_angles} dihedrals')

    curr_angles = np.array([dihedral.value(atoms) for dihedral in dihedrals])

    if all(dihedral.ring_n == 4 for dihedral in dihedrals):
        ideal_angles = np.array([0.0])

    elif all(dihedral.ring_n == 5 for dihedral in dihedrals):
        # 0, 40ª
        ideal_angles = np.array([0.0, 0.698132])[:n_angles]

    elif all(dihedral.ring_n == 6 for dihedral in dihedrals):
        # ±60º
        ideal_angles = np.array([1.0472, -1.0472, 1.0472])[:n_angles]

    elif all(dihedral.ring_n == 7 for dihedral in dihedrals):
        ideal_angles = np.array([1.2, -0.9, -0.6, 1.6])[:n_angles]

    else:
        raise NotImplementedError

    # Choose the ∆ϕs that are closest to the current values
    if (np.linalg.norm(curr_angles - ideal_angles)
            < np.linalg.norm(curr_angles + ideal_angles)):
        ideal_angles *= -1

    return curr_angles - ideal_angles
