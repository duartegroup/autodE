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


cdef calc_forces(int n_atoms, array forces, array coords, int[:, :] bond_matrix,
                double k, double[:, :] d0, double c):

    cdef int i, j
    cdef double delta_x = 0.0
    cdef double delta_y = 0.0
    cdef double delta_z = 0.0

    cdef double d = 0.0
    cdef double repulsion = 0.0
    cdef double bonded = 0.0
    cdef double fixed = 0.0

    # Zero the forces
    for i in range(3*n_atoms):
        forces.data.as_doubles[i] = 0.0

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i > j:
                delta_x = coords.data.as_doubles[3*j] - coords.data.as_doubles[3*i]
                delta_y = coords.data.as_doubles[3*j+1] - coords.data.as_doubles[3*i+1]
                delta_z = coords.data.as_doubles[3*j+2] - coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

                repulsion = -4.0 * c / pow(d, 6)
                forces.data.as_doubles[3*i] +=  repulsion * delta_x
                forces.data.as_doubles[3*i+1] +=  repulsion * delta_y
                forces.data.as_doubles[3*i+2] +=  repulsion * delta_z

                if bond_matrix[i][j] == 1:
                    bonded = 2.0 * k * (1.0 - d0[i][j]/d)
                    forces.data.as_doubles[3*i] += bonded * delta_x
                    forces.data.as_doubles[3*i+1] += bonded * delta_y
                    forces.data.as_doubles[3*i+2] += bonded * delta_z

                elif bond_matrix[i][j] == 2:
                    fixed = 2.0 * 100000 * (1.0 - d0[i][j]/d)
                    forces.data.as_doubles[3*i] += fixed * delta_x
                    forces.data.as_doubles[3*i+1] += fixed * delta_y
                    forces.data.as_doubles[3*i+2] += fixed * delta_z

    return forces


cdef calc_lambda(int n_atoms, array vel, float temp0):

    cdef double sum_mod_vel = 0.0
    cdef int i
    cdef double v_x, v_y, v_z

    for i in range(n_atoms):
        v_x = vel.data.as_doubles[3*i]
        v_y = vel.data.as_doubles[3*i+1]
        v_z = vel.data.as_doubles[3*i+2]
        sum_mod_vel += v_x*v_x + v_y*v_y + v_z*v_z

    cdef double temp = sum_mod_vel / (3.0 * <float>n_atoms)

    return sqrt(temp0 / temp)


def print_traj_point(py_xyzs, coords, traj_name):

    py_coords = np.asarray(coords).reshape(len(py_xyzs), 3)
    with open(traj_name, 'a') as traj_file:
        traj_file.write(str(len(py_xyzs))),
        traj_file.write('\n')
        traj_file.write('\n')
        xyzs = [[py_xyzs[m][0]] + py_coords[m].tolist() for m in range(len(py_xyzs))]
        for line in xyzs:
            traj_file.write('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(*line))
            traj_file.write('\n')

    return 0


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
                    energy += 100000 * pow((d - d0[i][j]), 2)
    return energy

def v(py_flat_coords, py_bonds, py_k, py_d0, py_c, py_fixed_bonds):

    py_n_atoms = int(len(py_flat_coords) / 3)
    cdef int n_atoms = py_n_atoms
    cdef int[:, :] bond_matrix = get_bond_matrix(n_atoms=py_n_atoms, bonds=py_bonds, fixed_bonds=py_fixed_bonds)
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0
    cdef double c = py_c

    cdef array coords, template = array('d')
    coords = clone(template, 3*n_atoms, False)

    cdef int i
    for i in range(3*n_atoms):
        coords[i] = py_flat_coords[i]

    return calc_energy(n_atoms, coords, bond_matrix, k, d0, c)


def do_md(py_xyzs, py_bonds, py_n_steps, py_temp, py_dt, py_k, py_d0, py_c, py_fixed_bonds):
    """
    Run an MD simulation under a potential:

    V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4

    where k and c are constants to be determined. Masses are all 1

    :param py_xyzs: (list(list)) e.g. [['C', 0.0, 0.0, 0.0], ...]
    :param py_bonds: (list(tuples)) defining which atoms are bonded together
    :param py_n_steps: (int) number of MD steps to do
    :param py_temp: (float) reduced temperature to run the dynamics (1 is fine)
    :param py_dt: (float) ∆t to use in the velocity verlet
    :param py_k: (float) harmonic force constant
    :param d0: (np.array) matrix of ideal bond lengths
    :param c: (float) strength of the repulsive term
    :param py_fixed_bonds: (list(tuples)) defining which atoms have fixed separations together
    :return: np array of coordinates
    """


    cdef int n_atoms = len(py_xyzs)

    py_coords = [xyz_line[1:4] for xyz_line in py_xyzs]
    py_flat_coords = [mu for coord in py_coords for mu in coord]

    # n_atoms x n_atoms array
    cdef int[:, :] bond_matrix = get_bond_matrix(n_atoms=len(py_xyzs), bonds=py_bonds, fixed_bonds=py_fixed_bonds)

    # Paramters for the MD simulation
    cdef int n_steps = py_n_steps
    cdef double temp0 = py_temp
    cdef double dt = py_dt
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0
    cdef double c = py_c

    # System intial time, velocities and accelteration
    cdef double t = 0.0
    cdef double temp_scale = 1.0
    cdef array coords, vel, a, forces, template = array('d')
    coords = clone(template, 3*n_atoms, False)
    vel = clone(template, 3*n_atoms, False)
    a = clone(template, 3*n_atoms, False)
    forces = clone(template, 3*n_atoms, False)

    # Set up c iterators
    cdef int i, n

    # Initialise arrays
    for i in range(3*n_atoms):
        vel[i] = 5.0 * np.random.normal()
        a[i] = 0.0
        forces[i] = 0.0
        coords[i] = 3 * np.random.normal()

    a = calc_forces(n_atoms, forces, coords, bond_matrix, k, d0, c)

    # traj_name = 'traj' + str(np.random.normal()) + '.xyz'

    for n in range(n_steps):

        t += dt
        for i in range(3 * n_atoms):
            coords.data.as_doubles[i] += dt * vel.data.as_doubles[i] + 0.5 * dt * dt * a.data.as_doubles[i]
            vel.data.as_doubles[i] += a.data.as_doubles[i] * dt

        a = calc_forces(n_atoms, forces, coords, bond_matrix, k, d0, c)

        temp_scale = calc_lambda(n_atoms, vel, temp0)
        for i in range(3 * n_atoms):
            vel.data.as_doubles[i] *= temp_scale

        # print_traj_point(py_xyzs, coords, traj_name)

    return np.asarray(coords).reshape(len(py_xyzs), 3)
