import pytest
import numpy as np
from autode.pes.relaxed import RelaxedPESnD
from autode.pes.pes_nd import Energies, PESnD


class TestPESnd(PESnD):

    __test__ = False

    def _default_keywords(self, method):
        return None

    def _calculate(self) -> None:
        raise NotImplementedError


def test_point_list_1d():

    pes = TestPESnd(rs={(0, 1): (1.0, 2.0, 3)})
    assert pes.ndim == 1
    assert list(pes._points()) == [(0,), (1,), (2,)]


def test_point_list_2d():

    pes = TestPESnd(rs={(0, 1): (1.0, 2.0, 2),
                           (1, 2): (1.0, 2.0, 2)})
    assert pes.ndim == 2
    assert pes.shape == (2, 2)

    assert list(pes._points()) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_point_list_non_square():

    pes = TestPESnd(rs={(0, 1): (1.0, 2.0, 2),
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

    pes = TestPESnd(rs={(0, 1): (1.0, 2.0, 3)})

    pes._energies = Energies(np.array([1.0, 0.01, 1.0]))
    assert len(list(pes._stationary_points())) == 1

    pes._energies = Energies(np.array([1.0, 1.1, 1.2]))
    assert len(list(pes._stationary_points())) == 0

    pes._energies = Energies(np.array([-1.0, -1.1, -1.2]))
    assert len(list(pes._stationary_points())) == 0

    pes._energies = Energies(np.array([-1.0, -1.0, -1.0]))
    assert len(list(pes._stationary_points())) == 0


def test_stationary_points_2d():

    def energy(x, y):
        return x * y - x**2 - x * y**2

    pes = TestPESnd(rs={(0, 1): (-1.5, 1.5, 10),
                        (1, 0): (-1.5, 1.5, 10)})

    pes._energies = Energies(energy(pes.r1, pes.r2))
    # pes.plot('tmp.pdf', interp_factor=0)
    # assert pes.shape == (50, 50)

    # Should have at least one stationary point. While in the
    # continuous surface there is 3, the finite surface may not have
    stat_points = list(pes._stationary_points())
    assert len(stat_points) > 0


def test_saddle_points_2d():

    def energy(x, y):
        return -x**2 + y**2

    pes = TestPESnd(rs={(0, 1): (-1.0, 1.0, 11),
                        (1, 0): (-1.0, 1.0, 11)})

    pes._energies = Energies(energy(pes.r1, pes.r2))
    pes.plot('tmp.pdf', interp_factor=0)

    # assert pes.shape == (50, 50)
    assert len(list(pes._stationary_points())) == 1

    # Should have at least one stationary point. While in the
    # continuous surface there is 3, the finite surface may not have
    points = list(pes._saddle_points())
    assert len(points) == 1

    p = points[0]

    # Saddle point should be close to (0, 0)
    assert np.isclose(pes.r1[p], 0.0, atol=0.1)
    assert np.isclose(pes.r2[p], 0.0, atol=0.1)
