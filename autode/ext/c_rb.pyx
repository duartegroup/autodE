# distutils: language = c++
# distutils: sources = [autode/ext/molecule.cpp, autode/ext/optimisers.cpp, autode/ext/potentials.cpp]
import numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from autode.ext.wrappers cimport Molecule, RBPotential, SDOptimiser


def minimised_rb_coords(py_coords,
                        py_bonded_matrix,
                        py_r0_matrix,
                        py_k_matrix,
                        py_c_matrix,
                        py_exponent):
    """
    Minimise a set of coordinates using a repulsion + bonded potential

    Arguments:
        py_coords:
        py_bonded_matrix:
        py_r0_matrix:
        py_k_matrix:
        py_c_matrix:
        py_exponent:

    Returns:
        (np.ndarray):
    """

    # Initialise a cpp Molecule
    cdef vector[double] coords = py_coords.flatten()
    cdef vector[bool_t] bonds = py_bonded_matrix.flatten()

    cdef Molecule molecule
    molecule = Molecule(coords, bonds)

    # and a repulsion + bonded potential
    cdef int exponent = py_exponent
    cdef vector[double] r0 = py_r0_matrix.flatten()
    cdef vector[double] k = py_k_matrix.flatten()
    cdef vector[double] c = py_c_matrix.flatten()

    cdef RBPotential potential
    potential = RBPotential(exponent, r0, k, c)

    # finally a steepest decent optimiser to use
    cdef SDOptimiser optimiser
    optimiser.run(potential, molecule, 500, 1E-4, 1E-1)

    py_coords = np.asarray(molecule.coords).reshape(-1, 3)
    return py_coords
