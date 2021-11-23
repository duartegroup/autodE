import numpy as np
import pytest
from autode.pes.relaxed import RelaxedPESnD


def test_point_list_1d():

    pes = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})
    assert pes.ndim == 1
    assert pes._points() == [(0,), (1,), (2,)]


def test_point_list_2d():

    pes = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2),
                           (1, 2): (1.0, 2.0, 2)})
    assert pes.ndim == 2
    assert pes.shape == (2, 2)

    assert pes._points() == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_point_list_non_square():

    pes = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2),
                           (1, 2): (1.0, 3.0, 3)})

    assert pes.ndim == 2 and pes.shape == (2, 3)

    points = pes._points()
    assert (points == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]
            or points == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)])

    assert np.isclose(pes.r1[1, 2], 2.0, atol=1E-10)
    assert np.isclose(pes.r2[1, 2], 3.0, atol=1E-10)


def test_closest_coordinates():

    pes = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})

    # Set a zero set of coordinates for 3 points
    # one atom with x, y, z coordinates
    pes._coordinates = np.zeros(shape=(3, 1, 3))

    # and the origin set of coordinates (1.0, 1.0, 1.0)
    pes._coordinates[0] = np.ones(shape=(1, 3))
    # with a defined energy
    pes._energies[0] = -1.0

    # So the closet point to the second (index (1,))
    # with an energy is the origin point
    assert np.allclose(pes._closest_coordinates(point=(1,)),
                       np.ones(shape=(1, 3)),
                       atol=1E-10)


def test_closest_coordinates_no_energy():

    pes = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})

    pes._coordinates = np.zeros(shape=(3, 1, 3))

    # Raises a runtime error if there is no suitable point
    with pytest.raises(RuntimeError):
        pes._closest_coordinates(point=(1,))

