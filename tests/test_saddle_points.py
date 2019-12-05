from autode import saddle_points
import numpy as np


def test_root_finder():
    # 3x^2 + 2y^2
    coeff_mat = np.array([[0, 0, 2], [0, 0, 0], [3, 0, 0]])
    # grads at (2,3)
    dx, dy = saddle_points.root_finder((2, 3), coeff_mat)
    assert dx == 12
    assert dy == 12


def test_calc_delta():
    # 3x^3 + 2y^3 + x^2y^2
    coeff_mat = np.array(
        [[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 1, 0], [3, 0, 0, 0]])
    assert saddle_points.calc_delta(coeff_mat, (2, 3)) == 1800


def test_poly2d_saddlepoints():
    # x^2 - y^2
    coeff_mat = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
    saddles = saddle_points.poly2d_saddlepoints(coeff_mat)
    assert type(saddles) == list
    assert -0.005 < saddles[0][0] < 0.005
    assert -0.005 < saddles[0][1] < 0.005
