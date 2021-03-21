# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.math cimport cos, sin, sqrt
import numpy as np


cdef void update_rotation_matrix(double [3][3] rot_mat,
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


cpdef rotate(py_coords,
             py_angles,
             py_axes,
             py_rot_idxs,
             py_origins):
    """
    Rotate coordinates by a set of dihedral angles each around an axis placed
    at an origin
    
    Arguments:
        py_coords: 
        py_angles: 
        py_axes: 
        py_rot_idxs: 
        py_origins: 
    
    Returns:
        (np.ndarray): Rotated coordinates
    """

    cdef int n_angles = len(py_angles)
    cdef int n_atoms = len(py_coords)

    cdef double [:, :] coords = py_coords

    cdef double [:] angles = py_angles
    cdef int [:, :] axes = py_axes
    cdef int [:] origins = py_origins
    cdef int [:, :] rot_idxs = py_rot_idxs

    cdef double [3][3] rot_mat
    cdef double [3] axis
    cdef double [3] origin

    cdef int i, j, k
    cdef double x, y, z

    for i in range(n_angles):

        # Define the origin and axis, the latter is normalised in
        # update_rotation_matrix
        for k in range(3):
            origin[k] = coords[origins[i], k]
            axis[k] = coords[axes[i, 0], k] - coords[axes[i, 1], k]

        update_rotation_matrix(rot_mat, axis=axis, theta=angles[i])

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

    return np.array(coords)
