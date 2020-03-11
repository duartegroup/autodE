import numpy as np
from autode import geom


def test_are_coords_reasonable():

    good_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert geom.are_coords_reasonable(coords=good_coords) is True

    bad_coords1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    assert geom.are_coords_reasonable(coords=bad_coords1) is False

    bad_coords2 = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [2.0, 0.0, 0.0],])
    assert geom.are_coords_reasonable(coords=bad_coords2) is False
