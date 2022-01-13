import numpy as np
from autode.geom import proj, rotate_columns
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.coordinates.primitives import Distance, InverseDistance
from autode.opt.coordinates.internals import PIC, InverseDistances


def test_joint_primitives_values():
    pic = PIC(InverseDistance(0, 1), InverseDistance(0, 2), Distance(1, 2))

    x = CartesianCoordinates([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.5, 0.5, 0.0]])

    # Ensure the values are as expected
    expected_q = [1.0, 1/np.sqrt(0.5**2 + 0.5**2), np.sqrt(0.5**2 + 0.5 **2)]
    assert np.allclose(pic(x), expected_q)


def test_proj_3_atom():
    """Project along the 3rd component (distance)"""

    x = CartesianCoordinates([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.5, 0.5, 0.0]])

    pic = PIC(InverseDistance(0, 1), InverseDistance(0, 2), Distance(1, 2))
    q = pic(x)

    G = np.matmul(pic.B, pic.B.T)

    eigvals, U = np.linalg.eigh(G)

    # Eigenvalues should all be strongly positive as the primitives span the
    # space without any redundancy
    assert np.all(eigvals > 1E-6)

    # Choose the first vector to have a unit component in Distance(1, 2)
    # i.e. put all the DIC weight into that primitive
    u1 = np.array([0.0, 0.0, 1.0])

    v1 = U[:, 1]
    u2 = v1 - proj(u1, v1)
    u2 /= np.linalg.norm(u2)

    v2 = U[:, 2]
    u3 = v2 - proj(u1, v2) - proj(u2, v2)
    u3 /= np.linalg.norm(u3)

    rotU = np.stack((u1, u2, u3), axis=-1)

    # Check the geom function does the same thing
    assert np.allclose(rotU,
                       rotate_columns(U, 2))

    # Matrix should still be orthonormal
    assert np.allclose(np.dot(rotU, rotU.T), np.eye(3))


def test_double_rot():
    """Test rotation can set two columns to unit vectors"""

    x = CartesianCoordinates([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.5, 0.5, 0.0],
                              [0.0, 0.1, 2.0]])

    pic = PIC(InverseDistance(0, 1),
              InverseDistance(0, 2),
              InverseDistance(1, 3),
              InverseDistance(2, 3),
              Distance(1, 2),
              Distance(0, 3))
    q = pic(x)

    G = np.matmul(pic.B, pic.B.T)
    eigvals, U = np.linalg.eigh(G)

    # No redundancy
    assert np.all(eigvals > 1E-6)

    arr = rotate_columns(U, 4, 5)

    # Should be 2 columns that have only 1 unity component, thus sum to 1
    assert sum(np.isclose(np.sum(arr[:, i]), 1.0) for i in range(6)) == 2

    assert np.allclose(np.dot(arr, arr.T), np.eye(len(arr)))


def test_tmp():
    x = CartesianCoordinates([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.5, 0.5, 0.0],
                              [0.0, 0.1, 2.0],
                              [0.1, 0.5, -0.3]])

    pic = InverseDistances()
    q = pic(x)

    # Should be redundant and have more primitives than 3N-6 DOF
    assert len(pic) > 3*5 - 6

    G = np.matmul(pic.B, pic.B.T)
    eigvals, U = np.linalg.eigh(G)

    v = U[:, np.where(np.abs(eigvals) > 1E-10)[0]]

    # Removing vectors with small eigenvalues
    assert v.shape != U.shape

    # Should not raise an exception for a non-symmetric matrix
    _ = rotate_columns(v, 1)
