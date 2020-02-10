# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from cpython.array cimport array, clone
from libc.math cimport sqrt, pow, exp, log
import numpy as np


def get_bond_matrix(n_atoms, bonds):

    bond_matrix = np.zeros((n_atoms, n_atoms), dtype=np.intc)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if (i, j) in bonds or (j, i) in bonds:
                bond_matrix[i, j] = 1

    return bond_matrix


cdef calc_forces(int n_solute_atoms, int n_solvent_atoms, array forces, array solute_coords, array solvent_coords, int[:, :] bond_matrix, array solute_charges, array solvent_charges, double k, double[:, :] d0):

    cdef int i, j
    cdef double delta_x = 0.0
    cdef double delta_y = 0.0
    cdef double delta_z = 0.0

    cdef double d = 0.0
    cdef double repulsion = 0.0
    cdef double bonded = 0.0

    cdef double min_d = 100.0
    cdef double min_dx = 0.0
    cdef double min_dy = 0.0
    cdef double min_dz = 0.0

    # Zero the forces
    for i in range(3*n_solvent_atoms):
        forces.data.as_doubles[i] = 0.0

    for i in range(n_solvent_atoms):
        for j in range(n_solvent_atoms):
            if i > j:
                delta_x = solvent_coords.data.as_doubles[3*j] - solvent_coords.data.as_doubles[3*i]
                delta_y = solvent_coords.data.as_doubles[3*j+1] - solvent_coords.data.as_doubles[3*i+1]
                delta_z = solvent_coords.data.as_doubles[3*j+2] - solvent_coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

                if bond_matrix[i][j] == 1:
                    bonded = 2.0 * k  * (1 - d0[i][j]/d)
                    forces.data.as_doubles[3*i] += bonded * delta_x
                    forces.data.as_doubles[3*i+1] += bonded * delta_y
                    forces.data.as_doubles[3*i+2] += bonded * delta_z            

                repulsion = - 36 / pow(d, 6) - (solvent_charges.data.as_doubles[i] * solvent_charges.data.as_doubles[j] / pow(d, 3))
                forces.data.as_doubles[3*i] += repulsion * delta_x
                forces.data.as_doubles[3*i+1] += repulsion * delta_y
                forces.data.as_doubles[3*i+2] += repulsion * delta_z

        for j in range(n_solute_atoms):
            delta_x = solute_coords.data.as_doubles[3*j] - solvent_coords.data.as_doubles[3*i]
            delta_y = solute_coords.data.as_doubles[3*j+1] - solvent_coords.data.as_doubles[3*i+1]
            delta_z = solute_coords.data.as_doubles[3*j+2] - solvent_coords.data.as_doubles[3*i+2]
            d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)
            if d < min_d:
                min_d = d
                min_dx = delta_x
                min_dy = delta_y
                min_dz = delta_z

            repulsion = - 12 / pow(d, 6) - (solvent_charges.data.as_doubles[i] * solute_charges.data.as_doubles[j] / pow(d, 3))
            forces.data.as_doubles[3*i] += repulsion * delta_x
            forces.data.as_doubles[3*i+1] += repulsion * delta_y
            forces.data.as_doubles[3*i+2] += repulsion * delta_z

        repulsion = 0.5*log(3*min_d)/min_d
        forces.data.as_doubles[3*i] += repulsion * min_dx
        forces.data.as_doubles[3*i+1] += repulsion * min_dy
        forces.data.as_doubles[3*i+2] += repulsion * min_dz

        min_d = 100.0

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

cdef calc_energy(int solute_n_atoms, int solvent_n_atoms, array solute_coords, array solvent_coords, int[:, :] bond_matrix, array solute_charges, array solvent_charges, double k, double[:, :] d0):

    cdef int i, j
    cdef double delta_x = 0.0
    cdef double delta_y = 0.0
    cdef double delta_z = 0.0

    cdef double d = 0.0
    cdef double min_d = 100.0
    cdef double repulsion = 0.0
    cdef double bonded = 0.0

    cdef double energy = 0.0

    for i in range(solvent_n_atoms):
        for j in range(solvent_n_atoms):
            if i < j:
                delta_x = solvent_coords.data.as_doubles[3*j] - solvent_coords.data.as_doubles[3*i]
                delta_y = solvent_coords.data.as_doubles[3*j+1] - solvent_coords.data.as_doubles[3*i+1]
                delta_z = solvent_coords.data.as_doubles[3*j+2] - solvent_coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)

                if bond_matrix[i][j] == 1:
                    energy += k * pow((d - d0[i][j]), 2)

                energy += 9 / pow(d, 4) + (solvent_charges.data.as_doubles[i] * solvent_charges.data.as_doubles[j])/ d
        
        for j in range(solute_n_atoms):
                delta_x = solute_coords.data.as_doubles[3*j] - solvent_coords.data.as_doubles[3*i]
                delta_y = solute_coords.data.as_doubles[3*j+1] - solvent_coords.data.as_doubles[3*i+1]
                delta_z = solute_coords.data.as_doubles[3*j+2] - solvent_coords.data.as_doubles[3*i+2]
                d = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z)
                if d < min_d:
                    min_d = d
                energy += 3 / pow(d, 4) + (solvent_charges.data.as_doubles[i] * solute_charges.data.as_doubles[j])/ d

        energy += 0.5*(min_d * log(3*min_d) - min_d)

        min_d = 100.0

    return energy

def v(py_flat_solvent_coords, py_flat_solute_coords, py_solvent_bonds, py_solute_charges, py_solvent_charges, py_k, py_d0):

    cdef int solvent_n_atoms = len(py_flat_solvent_coords) / 3
    cdef int solute_n_atoms = len(py_flat_solute_coords) / 3
    cdef int[:, :] bond_matrix = get_bond_matrix(n_atoms=solvent_n_atoms, bonds=py_solvent_bonds)
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0

    cdef array solvent_coords, solute_coords, solute_charges, solvent_charges, template = array('d')
    solvent_coords = clone(template, 3*solvent_n_atoms, False)
    solute_coords = clone(template, 3*solute_n_atoms, False)
    solvent_charges = clone(template, solvent_n_atoms, False)
    solute_charges = clone(template, solute_n_atoms, False)

    cdef int i
    for i in range(3*solvent_n_atoms):
        solvent_coords[i] = py_flat_solvent_coords[i]
    for i in range(3*solute_n_atoms):
        solute_coords[i] = py_flat_solute_coords[i]
    for i in range(solvent_n_atoms):
        solvent_charges[i] = py_solvent_charges[i]
    for i in range(solute_n_atoms):
        solute_charges[i] = py_solute_charges[i]

    return calc_energy(solute_n_atoms, solvent_n_atoms, solute_coords, solvent_coords, bond_matrix, solute_charges, solvent_charges, k, d0)


def do_md(py_solute_xyzs, py_solvent_xyzs, py_solvent_bonds, py_solute_charges, py_solvent_charges, py_n_steps, py_temp, py_dt, py_k, py_d0):

    cdef int solvent_n_atoms = len(py_solvent_xyzs)
    cdef int solute_n_atoms = len(py_solute_xyzs)

    py_solvent_coords = [xyz_line[1:4] for xyz_line in py_solvent_xyzs]
    py_solvent_flat_coords = [mu for coord in py_solvent_coords for mu in coord]

    py_solute_coords = [xyz_line[1:4] for xyz_line in py_solute_xyzs]
    py_solute_flat_coords = [mu for coord in py_solute_coords for mu in coord]

    # n_atoms x n_atoms array
    cdef int[:, :] bond_matrix = get_bond_matrix(n_atoms=len(py_solvent_xyzs), bonds=py_solvent_bonds)

    # Paramters for the MD simulation
    cdef int n_steps = py_n_steps
    cdef double temp0 = py_temp
    cdef double dt = py_dt
    cdef double k = py_k
    cdef double[:, :] d0 = py_d0

    # Solvent initial time, velocities and acceleration
    cdef double temp_scale = 1.0
    cdef array solvent_coords, vel, a, forces, solute_coords, solvent_charges, solute_charges, template = array('d')
    solvent_coords = clone(template, 3*solvent_n_atoms, False)
    vel = clone(template, 3*solvent_n_atoms, False)
    a = clone(template, 3*solvent_n_atoms, False)
    forces = clone(template, 3*solvent_n_atoms, False)
    solute_coords = clone(template, 3*solute_n_atoms, False)
    solvent_charges = clone(template, solvent_n_atoms, False)
    solute_charges = clone(template, solute_n_atoms, False)

    # Set up c iterators
    cdef int i, n, j

    # Initialise arrays
    for i in range(3*solvent_n_atoms):
        vel[i] = 0#np.random.normal()
        a[i] = 0.0
        forces[i] = 0.0
        solvent_coords[i] = py_solvent_flat_coords[i]
    for i in range(3*solute_n_atoms):
        solute_coords[i] = py_solute_flat_coords[i]
    for i in range(solvent_n_atoms):
        solvent_charges[i] = py_solvent_charges[i]
    for i in range(solute_n_atoms):
        solute_charges[i] = py_solute_charges[i]

    a = calc_forces(solute_n_atoms, solvent_n_atoms, forces, solute_coords, solvent_coords, bond_matrix, solute_charges, solvent_charges, k, d0)

    traj_name = 'traj' + str(np.random.normal()) + '.xyz'

    for n in range(n_steps):

        for i in range(3 * solvent_n_atoms):
                solvent_coords.data.as_doubles[i] += dt * vel.data.as_doubles[i] + 0.5 * dt * dt * a.data.as_doubles[i]
                vel.data.as_doubles[i] += a.data.as_doubles[i] * dt

        a = calc_forces(solute_n_atoms, solvent_n_atoms, forces, solute_coords, solvent_coords, bond_matrix, solute_charges, solvent_charges, k, d0)

        temp_scale = calc_lambda(solvent_n_atoms, vel, temp0)
        for i in range(3 * solvent_n_atoms):
            vel.data.as_doubles[i] *= temp_scale

        #print_traj_point(py_solvent_xyzs, solvent_coords, traj_name)

    return np.asarray(solvent_coords).reshape(len(py_solvent_xyzs), 3)
