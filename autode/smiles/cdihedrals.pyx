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


cdef void _set_coords(int n_atoms,
                      double [:, :] coords,
                      double [:, :] coords_):
    """
    Set all values in a set of coordinates given another set
    
    Args:
        n_atoms (int): 
        coords (array): 
        coords_ (array): 
    """

    for i in range(n_atoms):
        for k in range(3):
            coords[i, k] = coords_[i, k]

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


cdef double _energy(int n_atoms,
                    double [:, :] coords,
                    int [:, :] rep_idxs,
                    int [:] pair_idxs,
                    double r0):
    """
    Calculate the pairwise repulsion between all atoms with
    
    V(X) = Σ'_ij 1 / r_ij^2  +  (r_pair - r0)^2
    
    if pair = (0, 0) then the harmonic component is skipped
    
    Arguments:
        n_atoms (int): 
        coords (array): 
        rep_idxs (array):

    Returns:
        (double)
    """
    cdef int i, j
    cdef double energy = 0.0
    cdef double r_sq

    # Repulsive term
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):

            if rep_idxs[i, j] == 0:
                continue

            r_sq = ((coords[i, 0] - coords[j, 0])**2 +
                    (coords[i, 1] - coords[j, 1])**2 +
                    (coords[i, 2] - coords[j, 2])**2)

            energy += 1.0 / r_sq

    # Harmonic term
    if pair_idxs[0] != 0 and pair_idxs[1] != 0:
        r_pair = sqrt((coords[pair_idxs[0], 0] - coords[pair_idxs[1], 0])**2 +
                      (coords[pair_idxs[0], 1] - coords[pair_idxs[1], 1])**2 +
                      (coords[pair_idxs[0], 2] - coords[pair_idxs[1], 2])**2)

        energy += (r_pair - r0)**2

    return energy


cpdef double _minimise(int n_angles,
                       int n_atoms,
                       double [:, :] coords,
                       double [:] dangles,
                       int [:, :] axes,
                       int [:] origins,
                       int [:, :] rot_idxs,
                       int [:, :] rep_idxs,
                       int [:] pair_idxs,
                       double r0):
    """Minimise the dihedral rotations using a simple repulsive potential
    and a gradient decent minimiser
    """
    cdef double eps = 1E-7          # Finite difference  (rad)
    cdef double tol = 1E-3          # Tolerance on dE    (a.u.)
    cdef double step_size = 0.05    # Initial step size  (rad)
    cdef int max_iters = 10000      # Maximum number of iterations

    cdef double prev_energy = 0.0
    cdef double curr_energy = _energy(n_atoms, coords, rep_idxs, pair_idxs, r0)
    cdef int iteration = 0

    # Memory view arrays for the gradient dV/dθ_i and the forwards and
    # backwards displaced coordinates used for the finite difference
    cdef double [:] grad = np.zeros(int(n_angles))
    cdef double [:, :] coords_h = np.zeros(shape=(int(n_atoms), 3))
    cdef double [:, :] coords_mh = np.zeros(shape=(int(n_atoms), 3))

    cdef int i, j

    # Minimise the squared repulsion
    while (curr_energy - prev_energy)**2 > tol and iteration < max_iters:

        prev_energy = curr_energy

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
            grad[i] = (_energy(n_atoms, coords_h, rep_idxs, pair_idxs, r0)
                       - _energy(n_atoms, coords_mh, rep_idxs, pair_idxs, r0)) / eps

            dangles[i] += eps / 2.0


        # Calculate the new change in angle, based on the steepest decent
        for i in range(n_angles):
            dangles[i] = -step_size * grad[i]


        # Apply the optimal rotations δθ
        _rotate(n_angles, n_atoms, coords, dangles, axes, origins, rot_idxs)

        curr_energy = _energy(n_atoms, coords, rep_idxs, pair_idxs, r0)
        iteration += 1

    return curr_energy


cpdef void _close_ring(double [:, :] coords,
                       int [:, :] axes,
                       int [:] origins,
                       int [:, :] rot_idxs,
                       int [:, :] rep_idxs,
                       int [:] close_idxs,
                       double r0,
                       int n_iters=10):
    """Close a ring using dihedral rotations """

    cdef int n_angles = len(axes)
    cdef int n_atoms = len(coords)

    cdef double min_energy = 99999.9
    cdef double energy
    cdef double [:, :] min_coords = np.zeros(shape=(int(n_atoms), 3))
    cdef double [:, :] prev_coords = np.zeros(shape=(int(n_atoms), 3))

    cdef double [:] angles = np.zeros(shape=(int(n_angles)))

    cdef int n, i, k

    _set_coords(n_atoms, prev_coords, coords)

    # Run n_iters minimisation's
    for n in range(n_iters):

        angles = np.random.uniform(-np.pi, np.pi, size=int(n_angles))

        # Get the energy from some random starting point
        energy = _minimise(n_angles, n_atoms, coords, angles, axes, origins,
                           rot_idxs, rep_idxs, close_idxs, r0)

        if energy < min_energy:
            _set_coords(n_atoms, min_coords, coords)

        _set_coords(n_atoms, coords, prev_coords)


    _set_coords(n_atoms, coords, min_coords)
    return



cpdef rotate(py_coords,
             py_angles,
             py_axes,
             py_rot_idxs,
             py_origins,
             minimise=False,
             py_rep_idxs=None,
             py_close_idxs=None,
             py_r0=None):
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
        
        py_rep_idxs (np.ndarray): shape = (n_atoms, n_atoms). Square bit matrix
                                  indexing the atoms which should be considered
                                  to be pairwise repulsive. 
                                  *Only the upper triangular portion is used*
                                  
        py_close_idxs (np.ndarray): shape = (2,) Pair of atoms that need to be
                                    seperated by approximately r0
                                    
        py_r0 (float | None): r0 used if py_close_idxs is not None
                                  
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

    # Consider all pairwise repulsions, as only the upper triangle is used
    # the whole matrix can be ones
    if py_rep_idxs is None:
        py_rep_idxs = np.ones(shape=(int(n_atoms), int(n_atoms)),
                              dtype='i4')

    cdef int [:, :] rep_idxs = py_rep_idxs

    # Pair of atom indexes that will close a ring, the distance between which
    # needs to be close to r0
    if py_close_idxs is None:
        py_close_idxs = np.zeros(2, dtype='i4')

    assert py_close_idxs.shape == (2,)
    cdef int [:] close_idxs = py_close_idxs

    cdef double r0 = py_r0 if py_r0 is not None else 0.0


    # Apply all the rotations in place
    if py_close_idxs is not None and minimise:
        _close_ring(coords, axes, origins, rot_idxs, rep_idxs, close_idxs, r0)

    elif minimise:
        _minimise(n_angles, n_atoms, coords, angles, axes, origins, rot_idxs,
                  rep_idxs, close_idxs, r0)

    else:
        _rotate(n_angles, n_atoms, coords, angles, axes, origins, rot_idxs)

    return np.array(coords)
