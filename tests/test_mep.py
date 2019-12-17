from autode import min_energy_pathway as mep
import numpy as np
from numpy.polynomial import polynomial


def test_get_neighbouring_points():
    twobytwo_neighbours = mep.get_neighbouring_points((0, 0), 2)
    assert type(twobytwo_neighbours) == list
    assert len(twobytwo_neighbours) == 3

    twobytwo_neighbours_two = mep.get_neighbouring_points((1, 1), 2)
    assert len(twobytwo_neighbours_two) == 3

    threebythree_neighbours = mep.get_neighbouring_points((1, 1), 3)
    assert len(threebythree_neighbours) == 8


def test_get_point_grid():
    r1 = [0, 1, 2]
    r2 = [0, 1, 2]
    assert mep.get_point_on_grid((0.1, 1.4), r1, r2) == (0, 1)


def test_get_mep():
    # x^2 - 5xy - y^2
    coeff_mat = np.array([[0, 0, -1], [0, -5, 0], [1, 0, 0]])
    r1 = np.linspace(-0.5, 0.5, 5)
    x, y = np.meshgrid(r1, r1)
    energy_grid = polynomial.polyval2d(x, y, coeff_mat).transpose()
    assert mep.get_mep(r1, r1, energy_grid, (0, 0)) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
