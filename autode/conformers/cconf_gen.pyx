# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from cpython.array cimport array, clone
from libc.math cimport sqrt, pow
import numpy as np


cdef calc_energy(int n_atoms, array coords, int[:, :] bond_matrix, double k, double[:, :] d0, double c,
                 int exponent):

    cdef int i, j
    cdef double delta_x = 0.0
    cdef double delta_y = 0.0
    cdef double delta_z = 0.0

    cdef double d = 0.0
    cdef double repulsion = 0.0
    cdef double bonded = 0.0

    cdef double energy = 0.0


    for i in range(n_atoms):
        for j in range(n_atoms):
            if i > j:
                delta_x = coords.data.as_doubles[3*j] - coords.data.as_doubles[3*i]
                delta_y = coords.data.as_doubles[3*j+1] - coords.data.as_doubles[3*i+1]
                delta_z = coords.data.as_doubles[3*j+2] - coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

                energy += c / pow(d, exponent)

                if bond_matrix[i][j] == 1:
                    energy += k * pow((d - d0[i][j]), 2)

                if bond_matrix[i][j] == 2:
                    energy += 10 * pow((d - d0[i][j]), 2)
    return energy

cdef calc_deriv(int n_atoms, array deriv, array coords, int[:, :] bond_matrix,
                double k, double[:, :] d0, double c, int exponent):

    cdef int i, j
    cdef double delta_x
    cdef double delta_y
    cdef double delta_z

    cdef double d
    cdef double repulsion
    cdef double bonded
    cdef double fixed

    exponent_minus_2 = exponent + 2

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                delta_x = coords.data.as_doubles[3*j] - coords.data.as_doubles[3*i]
                delta_y = coords.data.as_doubles[3*j+1] - coords.data.as_doubles[3*i+1]
                delta_z = coords.data.as_doubles[3*j+2] - coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

                repulsion = -exponent * c / pow(d, exponent_minus_2)
                deriv.data.as_doubles[3*i] += repulsion * delta_x
                deriv.data.as_doubles[3*i+1] += repulsion * delta_y
                deriv.data.as_doubles[3*i+2] += repulsion * delta_z

                if bond_matrix[i][j] == 1:
                    bonded = 2.0 * k * (1.0 - d0[i][j]/d)
                    deriv.data.as_doubles[3*i] += bonded * delta_x
                    deriv.data.as_doubles[3*i+1] += bonded * delta_y
                    deriv.data.as_doubles[3*i+2] += bonded * delta_z

                if bond_matrix[i][j] == 2:
                    fixed = 20.0 * (1.0 - d0[i][j]/d)
                    deriv.data.as_doubles[3*i] += fixed * delta_x
                    deriv.data.as_doubles[3*i+1] += fixed * delta_y
                    deriv.data.as_doubles[3*i+2] += fixed * delta_z

    return -np.array(deriv)


def dvdr(py_flat_coords, py_bond_matrix, py_k, py_d0, py_c, py_exponent,
         py_fixed_atoms):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef int[:, :] bond_matrix = py_bond_matrix
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0
    cdef double c = py_c
    cdef int i
    cdef exponent = py_exponent

    cdef array coords, template = array('d')
    coords = clone(template, 3*n_atoms, False)
    init_array = clone(template, 3*n_atoms, False)

    # Initalise arrays
    for i in range(3*n_atoms):
        init_array[i] = 0.0
        coords[i] = py_flat_coords[i]

    dvdr = calc_deriv(n_atoms, init_array, coords, bond_matrix, k, d0, c, exponent)

    # Zero the gradients for all the fixed atoms
    dvdr = dvdr.reshape(-1, 3)
    dvdr[py_fixed_atoms, :] = 0.0

    return dvdr.flatten()


def v(py_flat_coords, py_bond_matrix, py_k, py_d0, py_c, py_exponent, *args):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef int[:, :] bond_matrix = py_bond_matrix
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0
    cdef double c = py_c
    cdef exponent = py_exponent

    cdef array coords, template = array('d')
    coords = clone(template, 3*n_atoms, False)

    cdef int i
    for i in range(3*n_atoms):
        coords[i] = py_flat_coords[i]

    return calc_energy(n_atoms, coords, bond_matrix, k, d0, c, exponent)
