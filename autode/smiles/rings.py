import numpy as np
from time import time
from itertools import permutations
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from autode.log import logger
from cdihedrals import rotate


def minimise_ring_energy(atoms, dihedrals, close_idxs, r0, ring_idxs,
                         max_iterations=10):
    """
    For a set of atoms minimise the energy with respect to dihedral rotation,
    where the energy is

    V = (r_close - r0)^2 + Σ'_ij 1 / (r_ij - 1)^4

    where i, j are atoms in the ring and sum includes all unique pairs that
    are not bonded.

    Note: atoms will be modified in place with respect to the rotation

    --------------------------------------------------------------------------
    Arguments:
        atoms (list(autode.smiles.SMILESAtom)): Atoms, a subset of which need
                                                rotation

        dihedrals (autode.smiles.builder.Dihedrals): Dihedrals to rotate about

        close_idxs (tuple(int)): Atom indexes for the bond to close

        r0 (float): Ideal distance (Å) for the closed bond

        ring_idxs (list(int)): All atom indexes within the ring

    Keyword Arguments:
        max_iterations (int): Maximum number of times the optimisation calls
    """
    start_time = time()

    coords = np.array([a.coord for a in atoms], copy=True, dtype='f8')

    # and 'opposing' atom indexes to add a repulsive components to the energy
    rep_idxs_i, rep_idxs_j = [], []
    for idx_i, idx_j in permutations(ring_idxs, r=2):

        # Only consider non-bonded ring indexes
        if 1 < idx_j - idx_i or idx_j - idx_i < -1:
            rep_idxs_i.append(idx_i)
            rep_idxs_j.append(idx_j)

    # Minimise the rotation energy with respect to the dihedral angles
    dist_mat = distance_matrix(coords, coords)
    iteration = 0

    # Minimise such that there are no close pairwise distances < 0.7 Å,
    # while ignoring any atoms that remain at the origin (=0 distance)
    while (iteration == 0 or
           (iteration < max_iterations
            and np.any(np.logical_and(dist_mat > 0.01, dist_mat < 0.7)))):

        res = minimize(dihedral_rotations,
                       x0=init_dihedral_angles(dihedrals, atoms),
                       args=(coords, dihedrals, close_idxs, r0,
                             (rep_idxs_i, rep_idxs_j)),
                       method='CG',
                       tol=1E-2)

        # Apply the optimal set of dihedral rotations
        new_coords = rotate(py_coords=coords,
                            py_angles=res.x,
                            py_axes=dihedrals.axes,
                            py_rot_idxs=dihedrals.rot_idxs,
                            py_origins=dihedrals.origins)

        dist_mat = distance_matrix(new_coords, new_coords)
        iteration += 1

    # and set the new coordinates
    for i, atom in enumerate(atoms):
        atom.coord = new_coords[i]

    logger.info(f'Closed ring in {(time() - start_time)*1000:.2f} ms')
    return None


def dihedral_rotations(angles, coords, dihedrals, close_idxs, r0, rep_idxs):
    """
    Perform a set of dihedral rotations and calculate the 'energy', defined
    by a harmonic in the difference between the ideal X-Y distance (r0) and
    it's current value, where X, Y are the two atoms in the tuple close_idxs

    Arguments:
        angles (np.ndarray):  shape = (m,)
        coords (np.ndarray):  shape = (n, 3)
        dihedrals (autode.smiles.builder.Dihedrals): Dihedrals to rotate about
        close_idxs (tuple(int)):
        r0 (float):

    Keyword Arguments:
        rep_idxs (tuple(np.ndarray)): A pair (tuple) of index arrays to add
                                      pairwise repulsion between
    Returns:
        (float):
    """
    coords = rotate(py_coords=np.copy(coords),
                    py_angles=angles,
                    py_axes=dihedrals.axes,
                    py_rot_idxs=dihedrals.rot_idxs,
                    py_origins=dihedrals.origins)
    energy = 0

    """
    Simple pairwise repulsive energy across the atoms in the ring
    as V = Σ'_ij 1 / (r_ij - 1)^4
    where the -1Å penalises distances closer than 1 Å more strongly
    as the repulsion is high if a non-bonded pair of atoms is < 1Å 
    away across a ring
    """
    idxs_i, idxs_j = rep_idxs
    dists = np.linalg.norm(coords[idxs_i] - coords[idxs_j], axis=1)
    energy += np.sum(np.power(dists - 1, -4)) / len(idxs_i)

    idx_i, idx_j = close_idxs

    # E_HO = (r_ij - r_0)^2
    energy += (np.linalg.norm(coords[idx_i] - coords[idx_j]) - r0)**2

    return energy


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
        ideal_angles = np.array([1.2, -0.9, 0.6, -1.6])[:n_angles]

    else:
        # Use ±π/2, which should be close for larger rings

        ideal_angles = []
        for i in range(n_angles):
            sign = 1 if i % 2 == 0 else -1
            ideal_angles.append(sign * np.pi/2.0)

        ideal_angles = np.array(ideal_angles)

    # Choose the ∆ϕs that are closest to the current values
    if (np.linalg.norm(curr_angles - ideal_angles)
            < np.linalg.norm(curr_angles + ideal_angles)):
        ideal_angles *= -1

    return curr_angles - ideal_angles
