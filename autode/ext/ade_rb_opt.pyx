# distutils: language = c++
# distutils: sources = [autode/ext/src/dihedrals.cpp, autode/ext/src/molecule.cpp, autode/ext/src/optimisers.cpp, autode/ext/src/potentials.cpp, autode/ext/src/utils.cpp, autode/ext/src/points.cpp]
import numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from autode.ext.wrappers cimport Molecule, RBPotential, SDOptimiser


def opt_rb_coords(py_coords,
                  py_bonded_matrix,
                  py_r0_matrix,
                  py_k_matrix,
                  py_c_matrix,
                  py_exponent):
    """
    Minimise a set of coordinates using a repulsion + bonded potential

    Arguments:
        py_coords (np.ndarray): Initial coordinates. shape = (n_atoms, 3)
        py_bonded_matrix (np.ndarray(bool)): shape = (n_atoms, n_atoms)
        py_r0_matrix (np.ndarray): shape = (n_atoms, n_atoms)
        py_k_matrix (np.ndarray): shape = (n_atoms, n_atoms)
        py_c_matrix (np.ndarray): shape = (n_atoms, n_atoms)
        py_exponent (float):

    Returns:
        (np.ndarray): Optimised coordinates
    """
    cdef vector[double] coords = py_coords.flatten()

    # Initialise a molecule with the default null constructor, then create
    cdef Molecule molecule
    molecule = Molecule(coords)

    # and a repulsion + bonded potential
    cdef int exponent = py_exponent
    cdef vector[bool_t] bonds = py_bonded_matrix.flatten()
    cdef vector[double] r0 = py_r0_matrix.flatten()
    cdef vector[double] k = py_k_matrix.flatten()
    cdef vector[double] c = py_c_matrix.flatten()

    cdef RBPotential potential
    potential = RBPotential(exponent,    # c/r_ij^exponent
                            bonds,       # boolean array of where bonds are
                            r0,          # ideal bond lengths for all pairs
                            k,           # k(r-r_0)^2 for all pairs
                            c)           # repulsive coefficient for all pairs

    # finally a steepest decent optimiser to use
    cdef SDOptimiser optimiser
    optimiser.run(potential,
                  molecule,
                  500,                   # Maximum number of iterations
                  1E-6,                  # Tolerance on ∆E_k->k+1 for a step k
                  0.3)                   # Initial step size (Å)

    py_coords = np.asarray(molecule.coords).reshape(-1, 3)
    return py_coords
