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


def test_get_breaking_breaking_bond_atom_id_dist_dict():

    breaking_bond_atom_ids = (0, 1)
    bbond_atom_id_dist_dict = geom.get_breaking_bond_atom_id_dist_dict(xyz_list,
                                                                       bbond_atom_ids_list=[breaking_bond_atom_ids])
    assert 0.999 < bbond_atom_id_dist_dict[(0, 1)] < 1.001


def test_valid_mappings():

    linear_h4_xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0],
                          ['H', 2.0, 0.0, 0.0], ['H', 3.0, 0.0, 0.0]]
    h2_graph = mol_graphs.make_graph(xyz_list, n_atoms=len(xyz_list))
    h4_graph = mol_graphs.make_graph(linear_h4_xyz_list, n_atoms=len(linear_h4_xyz_list))

    valid_mappings = geom.get_valid_mappings_frags_to_whole_graph(whole_graph=h4_graph,
                                                                  frag1_graph=h2_graph,
                                                                  frag2_graph=h2_graph)
    assert len(valid_mappings) == 24
    assert len(valid_mappings[0]) == 2
    assert type(valid_mappings[0][0]) == dict


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
