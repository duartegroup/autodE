# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.math cimport cos, sin, sqrt
import numpy as np


cdef void _update_rotation_matrix(double [3][3] rot_mat,
                                 double [3] axis,
                                 double theta):
    """
    Set the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Arguments:
        rot_mat: Rotation matrix to update
        axis: Unit vector in 3D to rotate around
        theta: Angle in radians
        
    Returns:
        rotation matrix:
    """

    cdef double norm = sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    cdef double sin_theta = sin(theta/2.0)

    cdef double a = cos(theta/2.0)
    cdef double b = -axis[0] * sin_theta / norm
    cdef double c = -axis[1] * sin_theta / norm
    cdef double d = -axis[2] * sin_theta / norm

    rot_mat[0][0] = a*a+b*b-c*c-d*d
    rot_mat[0][1] = 2.0*(b*c+a*d)
    rot_mat[0][2] = 2.0*(b*d-a*c)

    rot_mat[1][0] = 2.0*(b*c-a*d)
    rot_mat[1][1] = a*a+c*c-b*b-d*d
    rot_mat[1][2] = 2.0*(c*d+a*b)

    rot_mat[2][0] = 2.0*(b*d+a*c)
    rot_mat[2][1] = 2.0*(c*d-a*b)
    rot_mat[2][2] = a*a+d*d-b*b-c*c

    return


cdef void _rotate(int n_angles,
                  int n_atoms,
                  double [:, :] coords,
                  double [:] angles,
                  int [:, :] axes,
                  int [:] origins,
                  int [:, :] rot_idxs):
    """
    Rotate a set of dihedral angles - see rotate() for docstring
    """
    cdef double[3][3] rot_mat
    cdef double[3] axis
    cdef double[3] origin

    cdef int i, j, k
    cdef double x, y, z

    for i in range(n_angles):

        # Define the origin and axis, the latter is normalised in
        # update_rotation_matrix
        for k in range(3):
            origin[k] = coords[origins[i], k]
            axis[k] = coords[axes[i, 0], k] - coords[axes[i, 1], k]

        _update_rotation_matrix(rot_mat, axis=axis, theta=angles[i])

        # Rotate the portion of the structure to rotate about the bonding axis
        for j in range(n_atoms):

            # Only rotate the required atoms (rot_idx == 1, rather than 0)
            if rot_idxs[i, j] != 1:
                continue

            for k in range(3):
                coords[j, k] -= origin[k]

            # Apply the rotation
            x = coords[j, 0]
            y = coords[j, 1]
            z = coords[j, 2]

            for k in range(3):
                coords[j, k] = (rot_mat[k][0] * x +
                                rot_mat[k][1] * y +
                                rot_mat[k][2] * z)

            # And shift back from the origin
            for k in range(3):
                coords[j, k] += origin[k]

    return


cdef double _repulsion(int n_atoms, double [:, :] coords):
    """
    Calculate the pairwise repulsion between all atoms with
    
    V(X) = Σ'_ij 1 / r_ij^2
    
    Args:
        n_atoms (int): 
        coords (array): 

    Returns:
        (double)
    """
    cdef int i, j
    cdef double repulsion = 0.0
    cdef double dist_sq

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):

            dist_sq = ((coords[i, 0] - coords[j, 0])**2 +
                       (coords[i, 1] - coords[j, 1])**2 +
                       (coords[i, 2] - coords[j, 2])**2)

            repulsion += 1.0 / dist_sq

    return repulsion


cpdef void _minimise(int n_angles,
                int n_atoms,
                double [:, :] coords,
                double [:] dangles,
                int [:, :] axes,
                int [:] origins,
                int [:, :] rot_idxs):
    """Minimise the dihedral rotations using a simple repulsive potential
    and a gradient decent minimiser
    """
    cdef double eps = 1E-7          # Finite difference  (rad)
    cdef double tol = 1E-3          # Tolerance on dE    (a.u.)
    cdef double step_size = 0.05    # Initial step size  (rad)
    cdef int max_iters = 10000      # Maximum number of iterations

    cdef double prev_repulsion = 0.0
    cdef double curr_repulsion = _repulsion(n_atoms, coords)
    cdef int iteration = 0

    # Memory view arrays for the gradient dV/dθ_i and the forwards and
    # backwards displaced coordinates used for the finite difference
    cdef double [:] grad = np.zeros(int(n_angles))
    cdef double [:, :] coords_h = np.zeros(shape=(int(n_atoms), 3))
    cdef double [:, :] coords_mh = np.zeros(shape=(int(n_atoms), 3))

    cdef int i, j

    # Minimise the squared repulsion
    while (curr_repulsion - prev_repulsion)**2 > tol and iteration < max_iters:

        prev_repulsion = curr_repulsion

        for i in range(n_angles):   # θ_i

            # Reset the temporary set of coordinates for the finite difference
            for j in range(n_atoms):
                for k in range(3):
                    coords_h[j, k] = coords[j, k]
                    coords_mh[j, k] = coords[j, k]

            # Calculate a finite difference gradient
            dangles[i] += eps / 2.0
            _rotate(n_angles, n_atoms, coords_h, dangles, axes, origins, rot_idxs)

            dangles[i] -= eps
            _rotate(n_angles, n_atoms, coords_mh, dangles, axes, origins, rot_idxs)

            # central difference approximation for the gradient
            # dV/dθ_i ~ (V(x_h) - V(x_mh) / 2h)
            # where x_h are the coordinates from a +h/2 rotation on θ_i
            # and x_mh are similar with -h/2
            grad[i] = (_repulsion(n_atoms, coords_h)
                       - _repulsion(n_atoms, coords_mh)) / eps

            dangles[i] += eps / 2.0


        # Calculate the new change in angle, based on the steepest decent
        for i in range(n_angles):
            dangles[i] = -step_size * grad[i]

        # Apply the optimal rotations δθ
        _rotate(n_angles, n_atoms, coords, dangles, axes, origins, rot_idxs)

        curr_repulsion = _repulsion(n_atoms, coords)
        iteration += 1

    return


cpdef rotate(py_coords,
             py_angles,
             py_axes,
             py_rot_idxs,
             py_origins,
             minimise=False):
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
    
    Returns:
        (np.ndarray): Rotated coordinates
    """

    cdef int n_angles = len(py_angles)
    cdef int n_atoms = len(py_coords)

    # Use memory views of the numpy arrays
    cdef double [:, :] coords = py_coords

    cdef double [:] angles = py_angles
    cdef int [:, :] axes = py_axes
    cdef int [:] origins = py_origins
    cdef int [:, :] rot_idxs = py_rot_idxs

    # Apply all the rotations in place
    if minimise:
        _minimise(n_angles, n_atoms, coords, angles, axes, origins, rot_idxs)

    else:
        _rotate(n_angles, n_atoms, coords, angles, axes, origins, rot_idxs)

    return np.array(coords)
