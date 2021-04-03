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


cdef void _set_coords(double [:, :] coords,
                      double [:, :] coords_):
    """
    Set all values in a set of coordinates given another set
    
    Args:
        n_atoms (int): 
        coords (array): 
        coords_ (array): 
    """
    cdef int i, k

    for i in range(coords.shape[0]):
        for k in range(3):
            coords[i, k] = coords_[i, k]

    return


cdef void _rotate(double [:, :] coords,
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

    for i in range(angles.shape[0]):

        # Define the origin and axis, the latter is normalised in
        # update_rotation_matrix
        for k in range(3):
            origin[k] = coords[origins[i], k]
            axis[k] = coords[axes[i, 0], k] - coords[axes[i, 1], k]

        _update_rotation_matrix(rot_mat, axis=axis, theta=angles[i])

        # Rotate the portion of the structure to rotate about the bonding axis
        for j in range(coords.shape[0]):

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


cdef double _rep_energy(double [:, :] coords,
                        int [:, :] rep_idxs):
    """
    Calculate the pairwise repulsion between all atoms with {rep_idxs}ij != 0
    using
    
    V(coords) = Σ'_ij 1 / r_ij^2 
            
    Arguments:
        coords (array): 
        rep_idxs (array):

    Returns:
        (double)
    """
    cdef int i, j
    cdef double energy = 0.0
    cdef double r_sq

    # Repulsive term
    for i in range(coords.shape[0]):
        for j in range(i+1, coords.shape[0]):

            if rep_idxs[i, j] == 0:
                continue


            r_sq = ((coords[i, 0] - coords[j, 0])**2 +
                    (coords[i, 1] - coords[j, 1])**2 +
                    (coords[i, 2] - coords[j, 2])**2)

            energy += 1.0 / r_sq

    return energy


cdef double _close_energy(double [:, :] coords,
                          int [:, :] rep_idxs,
                          int [:] pair_idxs,
                          double r0):

    """
    Calculate the energy for closing a ring
    
    V(coords) = Σ'_ij 1 / r_ij^2  +  c(r_pair - r0)^2
    
    where ij is an atom pair and r_pair is the distance between the two atom
    indexes in pair_idxs
    
    Arguments:
        n_atoms (int): 
        coords (array): 
        rep_idxs (array):
        pair_idxs (array):
        r0 (double):

    Returns:
        (double)
    """
    cdef double repulsive_energy = _rep_energy(coords, rep_idxs)

    # Harmonic term
    r_pair = sqrt((coords[pair_idxs[0], 0] - coords[pair_idxs[1], 0])**2 +
                  (coords[pair_idxs[0], 1] - coords[pair_idxs[1], 1])**2 +
                  (coords[pair_idxs[0], 2] - coords[pair_idxs[1], 2])**2)

    return repulsive_energy + 10 * (r_pair - r0)**2


cpdef double _minimise(double [:, :] coords,
                       double [:] dangles,
                       int [:, :] axes,
                       int [:] origins,
                       int [:, :] rot_idxs,
                       int [:, :] rep_idxs):
    """Minimise the dihedral rotations using a simple repulsive potential
    and a gradient decent minimiser
    """
    cdef double eps = 1E-7          # Finite difference  (rad)
    cdef double tol = 1E-3          # Tolerance on dE    (a.u.)
    cdef double step_size = 0.05    # Initial step size  (rad)
    cdef int max_iters = 10000      # Maximum number of iterations

    cdef int n_atoms = coords.shape[0]
    cdef int n_angles = dangles.shape[0]

    cdef double prev_energy = 0.0
    cdef double curr_energy = _rep_energy(coords, rep_idxs)
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
            _rotate(coords_h, dangles, axes, origins, rot_idxs)

            dangles[i] -= eps
            _rotate(coords_mh, dangles, axes, origins, rot_idxs)

            # central difference approximation for the gradient
            # dV/dθ_i ~ (V(x_h) - V(x_mh) / 2h)
            # where x_h are the coordinates from a +h/2 rotation on θ_i
            # and x_mh are similar with -h/2
            grad[i] = (_rep_energy(coords_h, rep_idxs)
                       - _rep_energy(coords_mh, rep_idxs)) / eps

            dangles[i] += eps / 2.0


        # Calculate the new change in angle, based on the steepest decent
        for i in range(n_angles):
            dangles[i] = -step_size * grad[i]


        # Apply the optimal rotations δθ
        _rotate(coords, dangles, axes, origins, rot_idxs)

        curr_energy = _rep_energy(coords, rep_idxs)
        iteration += 1

    return curr_energy


cpdef void _close_ring2(double [:, :] coords,
                       double [:, :] min_coords,
                       double [:, :] prev_coords,
                       double [:, :, :] angles,
                       int [:, :] axes,
                       int [:] origins,
                       int [:, :] rot_idxs,
                       int [:, :] rep_idxs,
                       int [:] close_idxs,
                       double r0):
    """Close a ring on two dihedrals minimising over the whole 2D space"""

    cdef double min_energy = 99999.9
    cdef double energy = 0.0
    cdef int i, j

    _set_coords(prev_coords, coords)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):

            _rotate(coords, angles[i, j], axes, origins, rot_idxs)
            energy = _close_energy(coords, rep_idxs, close_idxs, r0)

            if energy < min_energy:
                _set_coords(min_coords, coords)
                min_energy = energy

            _set_coords(coords, prev_coords)

    _set_coords(coords, min_coords)
    return


cpdef void _close_ring1(double [:, :] coords,
                       double [:, :] min_coords,
                       double [:, :] prev_coords,
                       double [:, :] angles,
                       int [:, :] axes,
                       int [:] origins,
                       int [:, :] rot_idxs,
                       int [:, :] rep_idxs,
                       int [:] close_idxs,
                       double r0):
    """
    Close a ring on a single dihedral angle by enumerating all values 
    in the angles matrix, which is a column vector of angles for this single
    dihedral i.e. rot_idxs should have shape (1, m) etc.
    """

    cdef double min_energy = 99999.9
    cdef double energy
    cdef int i

    for i in range(angles.shape[0]):

        _rotate(coords, angles[i], axes, origins, rot_idxs)
        energy = _close_energy(coords, rep_idxs, close_idxs, r0)

        if energy < min_energy:
            _set_coords(min_coords, coords)
            min_energy = energy

        # Reset the coordinates to their initial values
        _set_coords(coords, prev_coords)

    # Finally set the optimal coordinates
    _set_coords(coords, min_coords)
    return


def large_ring_fixed(py_ideal_angles):
    """
    Generate the indexes of dihedrals that will be fixed and not minimised to
    close a ring depending on the number of angles, also return the ideal
    angles. Always return 2 non-fixed dihedrals that can be minimised on

    Arguments:
        py_ideal_angles (list(float | None)):

    Returns:
        (tuple(np.ndarray, np.ndarray)):
    """
    assert len(py_ideal_angles) > 2

    n_angles = len(py_ideal_angles)
    fixed_angles, fixed_idxs = [], []

    for i, angle in enumerate(py_ideal_angles):
        if angle is not None:
            fixed_angles.append(angle)
            fixed_idxs.append(i)

        if n_angles - len(fixed_idxs) == 2:
            return (np.array(fixed_idxs, dtype='i4'),
                    np.array(fixed_angles, dtype='f8'))

    if n_angles == 3:                          # e.g. 6-membered saturated ring
        # Fix one angle at 60º
        return (np.array([0], dtype='i4'),
                np.array([1.0472], dtype='f8'))

    if n_angles == 4:                        # e.g. 7-membered saturated ring
        # Fix angles at -70º, +60º
        return (np.array([0, 1], dtype='i4'),
                np.array([-1.22173, 1.0472], dtype='f8'))

    if n_angles == 5:                        # e.g. 8-membered saturated ring
        # Fix angles at ±84º
        return (np.array([0, 1, 2], dtype='i4'),
                np.array([1.46608, -1.46608, 1.46608], dtype='f8'))


    # Fix indexes are at the start and end of the ring, so insert any
    # remaining in the middle
    fixed_idxs = [0, 1, n_angles-2, n_angles-1]

    # start from -70º, +60º on either end of the ring
    fixed_angles = [-1.48353, 1.27409, -1.48353, 1.27409]

    # TODO: make this better
    for i in range(n_angles - 2 - 4):   # 2 optimised, 4 set

        if i % 2 == 0:
            fixed_idxs[i//2] += 1
            fixed_idxs[i//2+1] += 1

            fixed_idxs.insert(0, i//2)
            fixed_angles.insert(0, np.pi if i > 1 else 1.2)

        else:
            fixed_idxs.insert(-2, n_angles-2-(i//2+1))
            fixed_angles.insert(-2, np.pi if i > 1 else 1.2)

    return np.array(fixed_idxs, dtype='i4'), np.array(fixed_angles, dtype='f8')


cpdef closed_ring_coords(py_coords,
                         py_curr_angles,
                         py_ideal_angles,
                         py_axes,
                         py_rot_idxs,
                         py_origins,
                         py_rep_idxs,
                         py_close_idxs,
                         py_r0):
    """
    Close a ring by altering dihedral angles to minimise the pairwise repulsion
    while also minimising the distance between the two atoms that will close
    the ring with a harmonic term 
    (r_close - r0)^2
    
    --------------------------------------------------------------------------
    Arguments:
        py_coords (np.ndarray): shape = (n_atoms, 3)  Atomic coordinates
        
        py_curr_angles (np.ndarray): shape = (m,)  Current dihedral angles
        
        py_ideal_angles (list(float | None)): List of length m with the ideal
                                              angles, if None then no ideal angle
        
        py_axes (np.ndarray): shape = (m, 2) Atom indexes for the axis atoms
                              
        py_rot_idxs (np.ndarray): shape = (m, n_atoms) Bit array for each angle
                                  with 1 if this atom should be rotated and 0
                                  otherwise
        
        py_origins (np.ndarray): shape = (m,) Atom indexes of the origin atoms
            
        py_rep_idxs (np.ndarray): shape = (n_atoms, n_atoms). Index pairs to
                                  use for repulsion if {py_rep_idxs}_ij == 1
                                  
        py_close_idxs (np.ndarray): shape = (2,) Pair of atoms that need to be
                                    separated by approximately r0
                                    
        py_r0 (float): Optimal distance between the close indexes
                                  
    Returns:
        (np.ndarray): Rotated coordinates
    """

    # Use memory views of the numpy arrays
    cdef double [:, :] coords = py_coords
    cdef double [:, :] min_coords = np.copy(coords)
    cdef double [:, :] prev_coords = np.copy(coords)

    cdef int [:, :] axes = py_axes
    cdef int [:] origins = py_origins

    cdef int [:, :] rot_idxs = py_rot_idxs
    cdef int [:, :] rep_idxs = py_rep_idxs

    cdef int [:] close_idxs = py_close_idxs
    cdef double r0 = py_r0

    cdef double [:] angles
    cdef double [:, :] angles1
    cdef double [:, :, :] angles2

    n_angles = len(py_curr_angles)

    if n_angles == 0:
        return py_coords   # Nothing to rotate

    # For a completely defined set of ideal angles e.g. in a benzene ring
    # then all there is to do is apply the rotations
    if all([angle is not None for angle in py_ideal_angles]):

        angles = np.array(py_ideal_angles, dtype='f8') - py_curr_angles
        _rotate(coords, angles, axes, origins, rot_idxs)

        return np.array(coords)

    # There are at least one dihedral to optimise on...

    if n_angles == 1:
        # Evenly spaced set of angles over [-π, π), excluding the final
        # point in the array as a -π and π rotations are the same
        py_angles = np.linspace(-np.pi, np.pi-0.314, num=19) - py_curr_angles
        angles1 = np.expand_dims(py_angles, axis=1)

        _close_ring1(coords, min_coords, prev_coords, angles1, axes, origins,
                     rot_idxs, rep_idxs, close_idxs, r0)

        return np.array(coords)

    # There are at least two dihedrals to optimise on...

    # Use a coarser grid for a 2D grid search, and create a 3D tensor of angles
    # {angles}_ij = [angle_1, angle_2]  where the first two dimensions will
    # be iterated through
    py_angles1d = np.linspace(-np.pi, np.pi-0.314, num=19)
    angles2 = np.stack(np.meshgrid(py_angles1d, py_angles1d), axis=2)

    if n_angles == 2:
        # Nothing special to be done - just minimise fully
        _close_ring2(coords, min_coords, prev_coords, angles2, axes, origins,
                     rot_idxs, rep_idxs, close_idxs, r0)

        return np.array(coords)

    fixed_idxs, fixed_angles = large_ring_fixed(py_ideal_angles)

    # Apply the fixed rotations on a subset of the dihedrals, which will not
    # be optimised
    axes = py_axes[fixed_idxs, :]
    origins = py_origins[fixed_idxs]
    rot_idxs = py_rot_idxs[fixed_idxs, :]
    angles = fixed_angles - py_curr_angles[fixed_idxs]

    _rotate(coords, angles, axes, origins, rot_idxs)

    # Get the 'inverse' of the index array (surely there's a better way)
    rotated_idxs = np.array([i for i in range(n_angles)
                             if i not in fixed_idxs], dtype='i4')

    # Populate slices of the two axes, origins and rotation indexes for
    # the dihedrals that will be rotated
    axes = py_axes[rotated_idxs, :]
    origins = py_origins[rotated_idxs]
    rot_idxs = py_rot_idxs[rotated_idxs, :]


    # and find the optimal closure over the full [-π, π) range in each angle
    _close_ring2(coords, min_coords, prev_coords, angles2, axes, origins,
                 rot_idxs, rep_idxs, close_idxs, r0)

    # then refine around the minimum on the coarse grid
    angles2 = np.stack(np.meshgrid(np.linspace(-0.15, 0.15, num=19),
                                   np.linspace(-0.15, 0.15, num=19)), axis=2)

    _close_ring2(coords, min_coords, prev_coords, angles2, axes, origins,
                 rot_idxs, rep_idxs, close_idxs, r0)

    return np.array(coords)


cpdef rotate(py_coords,
             py_angles,
             py_axes,
             py_rot_idxs,
             py_origins,
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
        
        py_rep_idxs (np.ndarray): shape = (n_atoms, n_atoms). Square bit matrix
                                  indexing the atoms which should be considered
                                  to be pairwise repulsive. 
                                  *Only the upper triangular portion is used*
                                  
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
        py_rep_idxs = np.ones(shape=(int(n_atoms), int(n_atoms)), dtype='i4')

    cdef int [:, :] rep_idxs = py_rep_idxs

    # Apply all the rotations in place
    if minimise:
        _minimise(coords, angles, axes, origins, rot_idxs, rep_idxs)

    else:
        _rotate(coords, angles, axes, origins, rot_idxs)

    return np.array(coords)
