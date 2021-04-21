# distutils: language = c++
# distutils: sources = [autode/ext/dihedrals.cpp, autode/ext/molecule.cpp, autode/ext/optimisers.cpp, autode/ext/potentials.cpp]
import numpy as np
from autode.ext.wrappers cimport (Molecule,
                                  Dihedral,
                                  RDihedralPotential,
                                  RRingDihedralPotential,
                                  SDDihedralOptimiser,
                                  GridDihedralOptimiser,
                                  SGlobalDihedralOptimiser)


cdef Molecule molecule_with_dihedrals(py_coords,
                                       py_axes,
                                       py_rot_idxs,
                                       py_origins,
                                       py_angles=None):
    """Generate a autode::Molecule with defined dihedral angles"""

    cdef Molecule molecule
    cdef Dihedral dihedral

    molecule = Molecule(py_coords.flatten())

    # Set a dihedral for each axis, with perhaps a defined angle
    for i in range(py_axes.shape[0]):
        dihedral = Dihedral(0 if py_angles is None else py_angles[i],
                            np.asarray(py_axes[i], dtype='i4'),
                            np.asarray(py_rot_idxs[i], dtype=bool),
                            py_origins[i])

        molecule._dihedrals.push_back(dihedral)

    return molecule


def rotate(py_coords,
           py_angles,
           py_axes,
           py_rot_idxs,
           py_origins,
           rep_exponent=2,
           minimise=False,
           py_rep_idxs=None):
    """
    Rotate coordinates by a set of dihedral angles each around an axis placed
    at an origin

    --------------------------------------------------------------------------
    Arguments:
        py_coords (np.ndarray): shape = (n_atoms, 3)  Atomic coordinates in 3D

        py_angles (np.ndarray): shape = (m,)  Angles in radians to rotate by

        py_axes (np.ndarray): shape = (m, 2) Atom indexes for the two atoms
                              defining the rotation axis, for each angle

        py_rot_idxs (np.ndarray): shape = (m, n_atoms) Bit array for each angle
                                  with 1 if this atom should be rotated and 0
                                  otherwise

        py_origins (np.ndarray): shape = (m,) Atom indexes of the origin for
                                 each rotation

    Keyword Arguments:
        minimise (bool): Should the coordinates be minimised?

        py_rep_idxs (np.ndarray): shape = (n_atoms, n_atoms). Square boolean
                                  matrix indexing the atoms which should be
                                  considered to be pairwise repulsive.
                                  *Only the upper triangular portion is used*

    Returns:
        (np.ndarray): Rotated coordinates
    """

    cdef Molecule molecule = molecule_with_dihedrals(py_coords=py_coords,
                                                     py_axes=py_axes,
                                                     py_rot_idxs=py_rot_idxs,
                                                     py_origins=py_origins,
                                                     py_angles=py_angles)
    molecule.rotate_dihedrals()

    cdef RDihedralPotential potential
    cdef SDDihedralOptimiser optimiser

    # Consider all pairwise repulsions, as only the upper triangle is used
    # the whole matrix can be True
    if py_rep_idxs is None:
        n_atoms = py_coords.shape[0]
        py_rep_idxs = np.ones(shape=(n_atoms, n_atoms), dtype=bool)

    if minimise:
        potential = RDihedralPotential(rep_exponent,           # 1/r_ij^exponent
                                       py_rep_idxs.flatten())  # Repulsive pairs

        optimiser.run(potential,
                      molecule,
                      50,              # Maximum number of iterations
                      1E-5,            # Tolerance on ∆E_k->k+1 for a step k
                      1.0)             # Initial step size (Å)

    py_coords = np.asarray(molecule.coords).reshape(-1, 3)
    return py_coords


cpdef closed_ring_coords(py_coords,
                         py_curr_angles,
                         py_ideal_angles,
                         py_axes,
                         py_rot_idxs,
                         py_origins,
                         py_rep_idxs,
                         py_close_idxs,
                         py_r0=1.5,
                         py_rep_exponent=2):
    """
    Close a ring by altering dihedral angles to minimise the pairwise repulsion
    while also minimising the distance between the two atoms that will close
    the ring with a harmonic term: V = (r_close - r0)^2

    --------------------------------------------------------------------------
    Arguments:
        py_coords (np.ndarray): shape = (n_atoms, 3)  Atomic coordinates

        py_curr_angles (np.ndarray): shape = (m,)  Current dihedral angles

        py_ideal_angles (list(float | None)): List of length m with the ideal
                                              angles, if None then no ideal angle

        py_axes (np.ndarray): shape = (m, 2) Atom indexes for the axis atoms

        py_rot_idxs (np.ndarray): shape = (m, n_atoms) Bit array for each angle
                                  with 1 if this atom should be rotated and 0
                                  otherwise

        py_origins (np.ndarray): shape = (m,) Atom indexes of the origin atoms

        py_rep_idxs (np.ndarray): shape = (n_atoms, n_atoms). Index pairs to
                                  use for repulsion if {py_rep_idxs}_ij == 1

        py_close_idxs (np.ndarray): shape = (2,) Pair of atoms that need to be
                                    separated by approximately r0

    Keyword Arguments:

        py_r0 (float): Optimal distance between the close indexes

        py_rep_exponent (int): Exponent to use for the repulsive part of the
                               potential

    Returns:
        (np.ndarray): Rotated coordinates
    """
    n_angles = py_curr_angles.shape[0]

    if n_angles == 0:
        return py_coords  # Nothing to rotate

    # For a completely defined set of ideal angles e.g. in a benzene ring
    # then all there is to do is apply the rotations
    if all([angle is not None for angle in py_ideal_angles]):
        py_angles = [ideal - curr
                     for ideal, curr in zip(py_ideal_angles, py_curr_angles)]

        return rotate(py_coords=py_coords,
                      py_angles=py_angles,
                      py_axes=py_axes,
                      py_rot_idxs=py_rot_idxs,
                      py_origins=py_origins)

    # There are at least one dihedral to optimise on...
    cdef Molecule molecule = molecule_with_dihedrals(py_coords,
                                                     py_axes,
                                                     py_rot_idxs,
                                                     py_origins)
    cdef RRingDihedralPotential potential
    cdef GridDihedralOptimiser grid_optimiser
    cdef SGlobalDihedralOptimiser optimiser

    potential = RRingDihedralPotential(py_rep_exponent,       # 1/r_ij^exponent
                                       py_rep_idxs.flatten(), # Repulsive pairs
                                       py_close_idxs,         # Closing atoms
                                       py_r0)                 # Distance

    if n_angles <= 3:
        grid_optimiser.run(potential,
                           molecule,
                           500,     # Maximum number of grid points
                           1E-5,    # SD tolerance on ∆E_k->k+1 for a step k
                           1.0)
    else:
        optimiser.run(potential,
                      molecule,
                      100 * n_angles,      # Number of total steps
                      1E-5,                # Final tolerance on SD minimisation
                      1.0)

    py_coords = np.asarray(molecule.coords).reshape(-1, 3)
    return py_coords
