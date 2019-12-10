from autode import saddle_points
import numpy as np
from numpy.polynomial import polynomial


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


def test_best_saddlepoints():
    # x^2 + 2x^2y + y^2 + xy^2
    coeff_mat = np.array([[0, 0, 1], [0, 0, 2], [1, 2, 0]])
    saddles = [(-1.29816, 5.65202), (-0.46735, -0.410054),
               (1.09885, -0.575299)]
    r1 = np.linspace(-2, 2, 10)
    r2 = np.linspace(7, -3, 10)
    xx, yy = np.meshgrid(r1, r2)
    energy_grid = polynomial.polyval2d(xx, yy, coeff_mat)
    best_saddle_output = saddle_points.best_saddlepoint(
        saddles, r1, r2, energy_grid)
    assert 1.09883 < best_saddle_output[0] < 1.09887
    assert -0.575301 < best_saddle_output[1] < -0.575297
    assert type(best_saddle_output[2]) == list

    assert saddle_points.best_saddlepoint(
        [(10, 10)], r1, r2, energy_grid) is None
