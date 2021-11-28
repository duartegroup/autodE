import numpy as np
import pytest
from autode.pes.relaxed import RelaxedPESnD
from autode.pes.pes_nd import Energies


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


def test_constraints_1d():

    pes = RelaxedPESnD(rs={(0, 1): (0.1, 0.3, 3)})

    consts = pes._constraints(point=(0,))
    assert len(consts) == 1
    assert np.isclose(consts[(0, 1)], 0.1, atol=1E-10)

    for i in range(3):
        consts = pes._constraints(point=(i,))
        assert np.isclose(consts[(0, 1)], 0.1*(i+1), atol=1E-10)


def test_invalid_constraints_1d():

    pes = RelaxedPESnD(rs={(0, 1): (0.1, 0.3, 3)})

    # Cannot determine constraints for a point not on the surface
    with pytest.raises(ValueError):
        pes._constraints(point=(-1,))

    with pytest.raises(ValueError):
        pes._constraints(point=(0, 0))

    with pytest.raises(ValueError):
        pes._constraints(point=(3,))


def test_stationary_points_1d():
    """For a set 1D PESs ensure the stationary points can be found"""

    pes = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})

    pes._energies = Energies(np.array([1.0, 0.01, 1.0]))
    assert len(list(pes._stationary_points())) == 1

    pes._energies = Energies(np.array([1.0, 1.1, 1.2]))
    assert len(list(pes._stationary_points())) == 0

    pes._energies = Energies(np.array([-1.0, -1.1, -1.2]))
    assert len(list(pes._stationary_points())) == 0

    pes._energies = Energies(np.array([-1.0, -1.0, -1.0]))
    assert len(list(pes._stationary_points())) == 0
