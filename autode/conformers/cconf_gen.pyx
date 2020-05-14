# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from cpython.array cimport array, clone
from libc.math cimport sqrt, pow
import numpy as np


def get_bond_matrix(n_atoms, bonds, fixed_bonds):

    bond_matrix = np.zeros((n_atoms, n_atoms), dtype=np.intc)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if (i, j) in bonds or (j, i) in bonds:
                bond_matrix[i, j] = 1
            if (i, j) in fixed_bonds or (j, i) in fixed_bonds:
                bond_matrix[i, j] = 2

    return bond_matrix

cdef calc_energy(int n_atoms, array coords, int[:, :] bond_matrix, double k, double[:, :] d0, double c):

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

                energy += c / pow(d, 4)

                if bond_matrix[i][j] == 1:
                    energy += k * pow((d - d0[i][j]), 2)

                if bond_matrix[i][j] == 2:
                    energy += 10 * pow((d - d0[i][j]), 2)
    return energy

cdef calc_deriv(int n_atoms, array deriv, array coords, int[:, :] bond_matrix,
                double k, double[:, :] d0, double c):

    cdef int i, j
    cdef double delta_x
    cdef double delta_y
    cdef double delta_z

    cdef double d
    cdef double repulsion
    cdef double bonded
    cdef double fixed

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                delta_x = coords.data.as_doubles[3*j] - coords.data.as_doubles[3*i]
                delta_y = coords.data.as_doubles[3*j+1] - coords.data.as_doubles[3*i+1]
                delta_z = coords.data.as_doubles[3*j+2] - coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

                repulsion = -4.0 * c / pow(d, 6)
                deriv.data.as_doubles[3*i] -= repulsion * delta_x
                deriv.data.as_doubles[3*i+1] -= repulsion * delta_y
                deriv.data.as_doubles[3*i+2] -= repulsion * delta_z

                if bond_matrix[i][j] == 1:
                    bonded = 2.0 * k * (1.0 - d0[i][j]/d)
                    deriv.data.as_doubles[3*i] -= bonded * delta_x
                    deriv.data.as_doubles[3*i+1] -= bonded * delta_y
                    deriv.data.as_doubles[3*i+2] -= bonded * delta_z

                if bond_matrix[i][j] == 2:
                    fixed = 2.0 * 10 * (1.0 - d0[i][j]/d)
                    deriv.data.as_doubles[3*i] -= fixed * delta_x
                    deriv.data.as_doubles[3*i+1] -= fixed * delta_y
                    deriv.data.as_doubles[3*i+2] -= fixed * delta_z

    return deriv


def dvdr(py_flat_coords, py_bonds, py_k, py_d0, py_c, py_fixed_bonds, py_td, py_pi):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef int[:, :] bond_matrix = get_bond_matrix(n_atoms=py_n_atoms, bonds=py_bonds, fixed_bonds=py_fixed_bonds)
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0
    cdef double c = py_c
    cdef int i

    cdef array coords, deriv_array, template = array('d')
    coords = clone(template, 3*n_atoms, False)
    deriv_array = clone(template, 3*n_atoms, False)

    # Initalise arrays
    for i in range(3*n_atoms):
        deriv_array[i] = 0.0
        coords[i] = py_flat_coords[i]

    py_n_td = int(len(py_td) / 4)
    py_n_pi = int(len(py_pi) / 4)
    cdef int n_td = py_n_td
    cdef int n_pi = py_n_pi
    cdef array td_stereo, pi_stereo, itemplate = array('i')
    td_stereo = clone(itemplate, 4*n_td, False)
    pi_stereo = clone(itemplate, 4*n_pi, False)
    for i in range(4*n_td):
        td_stereo[i] = py_td[i]
    for i in range(4*n_pi):
        pi_stereo[i] = py_pi[i]

    deriv_array = calc_deriv(n_atoms, deriv_array, coords, bond_matrix, k, d0, c)
    return calc_stereo_deriv(deriv_array, coords, td_stereo, n_td, pi_stereo, n_pi)


def v(py_flat_coords, py_bonds, py_k, py_d0, py_c, py_fixed_bonds, py_td, py_pi):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef int[:, :] bond_matrix = get_bond_matrix(n_atoms=py_n_atoms, bonds=py_bonds, fixed_bonds=py_fixed_bonds)
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0
    cdef double c = py_c

    cdef array coords, dtemplate = array('d')
    coords = clone(dtemplate, 3*n_atoms, False)

    cdef int i
    for i in range(3*n_atoms):
        coords[i] = py_flat_coords[i]

    py_n_td = int(len(py_td) / 4)
    py_n_pi = int(len(py_pi) / 4)
    cdef int n_td = py_n_td
    cdef int n_pi = py_n_pi
    cdef array td_stereo, pi_stereo, itemplate = array('i')
    td_stereo = clone(itemplate, 4*n_td, False)
    pi_stereo = clone(itemplate, 4*n_pi, False)
    for i in range(4*n_td):
        td_stereo[i] = py_td[i]
    for i in range(4*n_pi):
        pi_stereo[i] = py_pi[i]

    return calc_energy(n_atoms, coords, bond_matrix, k, d0, c) + calc_stereo_energy(coords, td_stereo, n_td, pi_stereo, n_pi)


cdef cross(array vec1, array vec2):

    cdef array cross_vec, template = array('d')
    cross_vec = clone(template, 3, False)

    cross_vec[0] = (vec1.data.as_doubles[1]*vec2.data.as_doubles[2]) - (vec1.data.as_doubles[2]*vec2.data.as_doubles[1])
    cross_vec[1] = (vec1.data.as_doubles[2]*vec2.data.as_doubles[0]) - (vec1.data.as_doubles[0]*vec2.data.as_doubles[2])
    cross_vec[2] = (vec1.data.as_doubles[0]*vec2.data.as_doubles[1]) - (vec1.data.as_doubles[1]*vec2.data.as_doubles[0])
    
    return cross_vec


cdef dot(array vec1, array vec2):

    cdef double dot_product = 0.0

    dot_product += vec1.data.as_doubles[0]*vec2.data.as_doubles[0]
    dot_product += vec1.data.as_doubles[1]*vec2.data.as_doubles[1]
    dot_product += vec1.data.as_doubles[2]*vec2.data.as_doubles[2]

    return dot_product


def calc_length(array vec):

    cdef double length = 0.0
    
    length += pow(vec.data.as_doubles[0], 2)
    length += pow(vec.data.as_doubles[1], 2)
    length += pow(vec.data.as_doubles[2], 2)
    length = sqrt(length)

    return length


cdef normalise(array vec):

    cdef double length = calc_length(vec)

    vec[0] = vec.data.as_doubles[0]/length
    vec[1] = vec.data.as_doubles[1]/length
    vec[2] = vec.data.as_doubles[2]/length

    return vec


cdef vec_subtraction(array vec1, array vec2):

    cdef array difference_vec, template = array('d')
    difference_vec = clone(template, 3, False)

    difference_vec[0] = vec1.data.as_doubles[0] - vec2.data.as_doubles[0]
    difference_vec[1] = vec1.data.as_doubles[1] - vec2.data.as_doubles[1]
    difference_vec[2] = vec1.data.as_doubles[2] - vec2.data.as_doubles[2]

    return difference_vec


cdef normed_difference(array coords, int atom1, array centre):

    cdef array normed_difference_vec, template = array('d')
    normed_difference_vec = clone(template, 3, False)

    normed_difference_vec[0] = coords.data.as_doubles[3*atom1] - centre.data.as_doubles[0]
    normed_difference_vec[1] = coords.data.as_doubles[3*atom1+1] - centre.data.as_doubles[1]
    normed_difference_vec[2] = coords.data.as_doubles[3*atom1+2] - centre.data.as_doubles[2]
    normalise(normed_difference_vec)

    return normed_difference_vec


def average(array coords, int atom1, int atom2, int atom3, int atom4):
    cdef array centre, template = array('d')
    centre = clone(template, 3, False)

    centre[0] = (coords.data.as_doubles[3*atom1] + coords.data.as_doubles[3*atom2] + coords.data.as_doubles[3*atom3] + coords.data.as_doubles[3*atom4])/4
    centre[1] = (coords.data.as_doubles[3*atom1+1] + coords.data.as_doubles[3*atom2+1] + coords.data.as_doubles[3*atom3+1] + coords.data.as_doubles[3*atom4+1])/4
    centre[2] = (coords.data.as_doubles[3*atom1+2] + coords.data.as_doubles[3*atom2+2] + coords.data.as_doubles[3*atom3+2] + coords.data.as_doubles[3*atom4+2])/4

    return centre


cdef calc_td_volume(array coords, int atom1, int atom2, int atom3, int atom4):

    cdef array centre = average(coords, atom1, atom2, atom3, atom4)

    cdef array scaled_coord1 = normed_difference(coords, atom1, centre)
    cdef array scaled_coord2 = normed_difference(coords, atom2, centre)

    cdef array vec1 = vec_subtraction(scaled_coord2, scaled_coord1)

    cdef array scaled_coord3 = normed_difference(coords, atom3, centre)
    cdef array vec2 = vec_subtraction(scaled_coord3, scaled_coord1)

    cdef array scaled_coord4 = normed_difference(coords, atom4, centre)
    cdef array vec3 = vec_subtraction(scaled_coord4, scaled_coord1)

    #vec2 x vec3
    cdef array cross_vec = cross(vec2, vec3)

    #vec1 . cross_vec
    return dot(vec1, cross_vec)


def calc_quadrilateral_area(array coords, int atom1, int atom2, int atom3):

    cdef array vec1, vec2, cross_vec, template = array('d')
    vec1 = clone(template, 3, False)
    vec2 = clone(template, 3, False)
    cross_vec = clone(template, 3, False)
    cdef double area = 0.0

    vec1[0] = coords.data.as_doubles[3*atom2] - coords.data.as_doubles[3*atom1]
    vec1[1] = coords.data.as_doubles[3*atom2+1] - coords.data.as_doubles[3*atom1+1]
    vec1[2] = coords.data.as_doubles[3*atom2+2] - coords.data.as_doubles[3*atom1+2]

    vec2[0] = coords.data.as_doubles[3*atom3] - coords.data.as_doubles[3*atom1]
    vec2[1] = coords.data.as_doubles[3*atom3+1] - coords.data.as_doubles[3*atom1+1]
    vec2[2] = coords.data.as_doubles[3*atom3+2] - coords.data.as_doubles[3*atom1+2]

    cross_vec = cross(vec1, vec2)
    
    area = pow(cross_vec.data.as_doubles[0], 2) + pow(cross_vec.data.as_doubles[1], 2) + pow(cross_vec.data.as_doubles[2], 2)

    return area


cdef check_pi_angle(array coords, int atom1, int atom2, int atom3, int atom4):
    cdef array centre = average(coords, atom1, atom2, atom3, atom4)

    cdef array scaled_coord1 = normed_difference(coords, atom1, centre)
    cdef array scaled_coord2 = normed_difference(coords, atom2, centre)
    cdef array vec1 = vec_subtraction(scaled_coord2, scaled_coord1)
    normalise(vec1)

    cdef array scaled_coord3 = normed_difference(coords, atom3, centre)
    cdef array scaled_coord4 = normed_difference(coords, atom4, centre)
    cdef array vec2 = vec_subtraction(scaled_coord4, scaled_coord3)
    normalise(vec2)

    return dot(vec1, vec2)


cdef calc_stereo_energy(array coords, array td_stereo, int n_td, array pi_stereo, int n_pi):

    cdef double volume = 0.0
    cdef double energy = 0.0

    cdef int i
    for i in range(n_td):
        #The signed volume of a tetrahedron tells us if the correct sterochemistry is achieved. For a tetrahedron with vertices v1, v2, v3, v4
        # the volume = (v2-v1).((v3-v1)x(v4-v1)). The order of vertices we choose and scaling such that the points are a distance of 1 from the
        # centre means this volume *should* be -16sqrt(3)/9 for the correct stereochemistry
        volume = calc_td_volume(coords, td_stereo[4*i], td_stereo[4*i+1], td_stereo[4*i+2], td_stereo[4*i+3])
        volume += 16*sqrt(3)/9
        energy += pow(volume,2)

    for i in range(n_pi):
        energy += calc_pi_stereo_energy(coords, pi_stereo[4*i], pi_stereo[4*i+1], pi_stereo[4*i+2], pi_stereo[4*i+3])
         
    return energy


cdef calc_pi_stereo_energy(array coords, int atom1, int atom2, int atom3, int atom4):
    cdef double pi_energy = 0.0

    #Alkenes should be flat .'. the volume should be zero
    cdef double volume = calc_td_volume(coords, atom1, atom2, atom3, atom4)
    pi_energy += pow(volume,2)

    # This ensures the correct stereochemistry
    cdef double angle = check_pi_angle(coords, atom1, atom2, atom3, atom4)
    angle += 1
    pi_energy += 5 * pow(angle, 2)

    return 0.001 * pi_energy


def stereo(py_flat_coords, py_td, py_pi):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef array coords, dtemplate = array('d')
    coords = clone(dtemplate, 3*n_atoms, False)
    cdef int i
    for i in range(3*n_atoms):
        coords[i] = py_flat_coords[i]

    py_n_td = int(len(py_td) / 4)
    py_n_pi = int(len(py_pi) / 4)
    cdef int n_td = py_n_td
    cdef int n_pi = py_n_pi
    cdef array td_stereo, pi_stereo, itemplate = array('i')
    td_stereo = clone(itemplate, 4*n_td, False)
    pi_stereo = clone(itemplate, 4*n_pi, False)
    for i in range(4*n_td):
        td_stereo[i] = py_td[i]
    for i in range(4*n_pi):
        pi_stereo[i] = py_pi[i]

    return calc_stereo_energy(coords, td_stereo, n_td, pi_stereo, n_pi)


cdef calc_stereo_deriv(array deriv, array coords, array td_stereo, int n_td, array pi_stereo, int n_pi):

    cdef double value = 0.0
    cdef double energy = 0.0
    cdef double shift_energy = 0.0

    cdef int i
    cdef int j
    cdef int k
    for i in range(n_td):
        value = calc_td_volume(coords, td_stereo[4*i], td_stereo[4*i+1], td_stereo[4*i+2], td_stereo[4*i+3])
        value += 16*sqrt(3)/9
        energy = pow(value,2)

        for j in range(4):
            for k in range(3):
                coords.data.as_doubles[3*td_stereo[4*i+j]+k] += 0.0001
                value = calc_td_volume(coords, td_stereo[4*i], td_stereo[4*i+1], td_stereo[4*i+2], td_stereo[4*i+3])
                value += 16*sqrt(3)/9
                shift_energy = pow(value,2)
                deriv.data.as_doubles[3*td_stereo[4*i+j]+k] += 10000*(shift_energy-energy)
                coords.data.as_doubles[3*td_stereo[4*i+j]+k] -= 0.0001


    for i in range(n_pi):
        energy = calc_pi_stereo_energy(coords, pi_stereo[4*i], pi_stereo[4*i+1], pi_stereo[4*i+2], pi_stereo[4*i+3])

        for j in range(4):
            for k in range(3):
                coords.data.as_doubles[3*pi_stereo[4*i+j]+k] += 0.0001
                shift_energy = calc_pi_stereo_energy(coords, pi_stereo[4*i], pi_stereo[4*i+1], pi_stereo[4*i+2], pi_stereo[4*i+3])
                deriv.data.as_doubles[3*pi_stereo[4*i+j]+k] += 10000*(shift_energy-energy)
                coords.data.as_doubles[3*pi_stereo[4*i+j]+k] -= 0.0001


    return np.array(deriv)


def dstereodr(py_flat_coords, py_td, py_pi):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef array coords, deriv_array, dtemplate = array('d')
    coords = clone(dtemplate, 3*n_atoms, False)
    deriv_array = clone(dtemplate, 3*n_atoms, False)
    cdef int i
    for i in range(3*n_atoms):
        deriv_array[i] = 0.0
        coords[i] = py_flat_coords[i]

    py_n_td = int(len(py_td) / 4)
    py_n_pi = int(len(py_pi) / 4)
    cdef int n_td = py_n_td
    cdef int n_pi = py_n_pi
    cdef array td_stereo, pi_stereo, itemplate = array('i')
    td_stereo = clone(itemplate, 4*n_td, False)
    pi_stereo = clone(itemplate, 4*n_pi, False)
    for i in range(4*n_td):
        td_stereo[i] = py_td[i]
    for i in range(4*n_pi):
        pi_stereo[i] = py_pi[i]

    return calc_stereo_deriv(deriv_array, coords, td_stereo, n_td, pi_stereo, n_pi)