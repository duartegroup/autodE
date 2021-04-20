# distutils: language = c++
# distutils: sources = [autode/ext/dihedrals.cpp, autode/ext/molecule.cpp, autode/ext/optimisers.cpp, autode/ext/potentials.cpp]
import numpy as np
from autode.ext.wrappers cimport (Molecule,
                                  Dihedral,
                                  RDihedralPotential,
                                  SDDihedralOptimiser)


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

    cdef Molecule molecule
    cdef Dihedral dihedral

    molecule = Molecule(py_coords.flatten())

    for i, py_angle in enumerate(py_angles):

        dihedral = Dihedral(py_angle,
                            np.asarray(py_axes[i], dtype='i4'),
                            np.asarray(py_rot_idxs[i], dtype=bool),
                            py_origins[i])

        molecule._dihedrals.push_back(dihedral)


    # Rotate and minimise, the latter 'globally'
    molecule.rotate_dihedrals()


    cdef RDihedralPotential potential
    cdef SDDihedralOptimiser optimiser

    # Consider all pairwise repulsions, as only the upper triangle is used
    # the whole matrix can be ones
    if py_rep_idxs is None:
        py_rep_idxs = np.ones(shape=(py_coords.shape[0],
                                     py_coords.shape[0]), dtype=bool)

    if minimise:
        potential = RDihedralPotential(rep_exponent,           # 1/r_ij^exponent
                                       py_rep_idxs.flatten())  # Repulsive pairs

        optimiser.run(potential,
                      molecule,
                      50,              # Maximum number of iterations
                      1E-5,            # Tolerance on ∆E_k->k+1 for a step k
                      1)               # Initial step size (Å)

    py_coords = np.asarray(molecule.coords).reshape(-1, 3)
    return py_coords
