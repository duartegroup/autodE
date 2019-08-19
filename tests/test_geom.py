from autode import geom
from autode import mol_graphs
import numpy as np

xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]


def test_xyz2coord():

    coord_list = geom.xyz2coord(xyz_list)

    assert type(coord_list) == np.ndarray
    assert type(coord_list[0]) == np.ndarray
    assert 0.99 < np.linalg.norm(coord_list[0] - coord_list[1]) < 1.001

    xyz_line = ['H', 0.0, 0.0, 0.0]
    coord = geom.xyz2coord(xyz_line)

    assert type(coord) == np.ndarray
    assert len(coord) == 3


def test_distance_matrix():

    distance_matix = geom.calc_distance_matrix(xyz_list)
    assert distance_matix.shape == (2, 2)
    assert distance_matix[0, 0] == 0.0


def test_rot_matix():

    axis = np.array([0.0, 0.0, 1.0])
    theta = np.pi                       # angle in radians

    rot_matix = geom.calc_rotation_matrix(axis, theta)
    point = np.array([1.0, 1.0, 1.0])
    rot_point = np.matmul(rot_matix, point)

    assert rot_matix.shape == (3, 3)
    assert -1.001 < rot_matix[0, 0] < -0.999
    assert -1.001 < rot_matix[1, 1] < -0.999
    assert 0.999 < rot_matix[2, 2] < 1.001
    assert -0.001 < rot_matix[0, 1] < 0.001

    assert -1.001 < rot_point[0] < -0.999
    assert -1.001 < rot_point[1] < -0.999
    assert 0.999 < rot_point[2] < 1.001
