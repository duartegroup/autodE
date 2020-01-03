from autode import geom
import numpy as np
from autode import molecule


xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]
xyz_line = ['H', 0.0, 0.0, 0.0]


def test_xyz2coord():

    coord_list = geom.xyz2coord(xyz_list)

    assert type(coord_list) == np.ndarray
    assert type(coord_list[0]) == np.ndarray
    assert 0.99 < np.linalg.norm(coord_list[0] - coord_list[1]) < 1.001

    coord = geom.xyz2coord(xyz_line)

    assert type(coord) == np.ndarray
    assert len(coord) == 3


def test_distance_matrix():

    distance_matrix = geom.calc_distance_matrix(xyz_list)
    assert distance_matrix.shape == (2, 2)
    assert distance_matrix[0, 0] == 0.0


def test_rot_matrix():

    axis = np.array([0.0, 0.0, 1.0])
    theta = np.pi                       # angle in radians

    rot_matrix = geom.calc_rotation_matrix(axis, theta)
    point = np.array([1.0, 1.0, 1.0])
    rot_point = np.matmul(rot_matrix, point)

    assert rot_matrix.shape == (3, 3)
    assert -1.001 < rot_matrix[0, 0] < -0.999
    assert -1.001 < rot_matrix[1, 1] < -0.999
    assert 0.999 < rot_matrix[2, 2] < 1.001
    assert -0.001 < rot_matrix[0, 1] < 0.001

    assert -1.001 < rot_point[0] < -0.999
    assert -1.001 < rot_point[1] < -0.999
    assert 0.999 < rot_point[2] < 1.001


def test_coords2xyzs():

    new_coords = [np.array([0.0, 0.0, 0.0]), np.array([0.9, 0.0, 0.0])]
    new_coords_single = np.array([0.0, 0.0, 1.0])

    new_xyzs = geom.coords2xyzs(new_coords, xyz_list)

    assert type(new_xyzs) == list
    assert type(new_xyzs[0]) == list
    assert 0.099 < (xyz_list[1][1] - new_xyzs[1][1]) < 0.101

    new_xyzs_single = geom.coords2xyzs(new_coords_single, xyz_line)

    assert type(new_xyzs_single) == list
    assert len(new_xyzs_single) == 4
    assert 0.999 < (new_xyzs_single[3] - xyz_line[3]) < 1.001


def test_neighbour_list():

    methane = molecule.Molecule(name='methane', smiles='C')
    neighbour_list = geom.get_neighbour_list(0, methane)
    assert len(neighbour_list) == 5
    assert type(neighbour_list) == list
    assert type(neighbour_list[0]) == str


def test_shifted_xyzs_linear_interp():

    new_xyzs = geom.get_shifted_xyzs_linear_interp(xyzs=xyz_list, bonds=[(0, 1)], final_distances=[0.7])
    new_coords = geom.xyz2coord(xyzs=new_xyzs)

    assert 0.69 < np.linalg.norm(new_coords[0] - new_coords[1]) < 0.71
