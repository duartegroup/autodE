import pytest
import numpy as np
from autode.pes.pes_nd import EnergyArray as Energies
from autode.pes.relaxed import RelaxedPESnD as PESnD
from .sample_pes import TestPES, harmonic_2d_pes


class RelaxedPESnD(PESnD):

    __test__ = False

    def __init__(self, species=None, rs=None):
        super(RelaxedPESnD, self).__init__(species=species, rs=rs)


def test_point_list_1d():

    pes = TestPES(rs={(0, 1): (1.0, 2.0, 3)})
    assert pes.ndim == 1
    assert list(pes._points()) == [(0,), (1,), (2,)]


def test_point_list_2d():

    pes = TestPES(rs={(0, 1): (1.0, 2.0, 2), (1, 2): (1.0, 2.0, 2)})
    assert pes.ndim == 2
    assert pes.shape == (2, 2)

    assert list(pes._points()) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_point_list_non_square():

    pes = TestPES(rs={(0, 1): (1.0, 2.0, 2), (1, 2): (1.0, 3.0, 3)})

    assert pes.ndim == 2 and pes.shape == (2, 3)

    points = pes._points()
    assert points == [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (1, 2),
    ] or points == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (1, 2),
    ]

    assert np.isclose(pes.r1[1, 2], 2.0, atol=1e-10)
    assert np.isclose(pes.r2[1, 2], 3.0, atol=1e-10)


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
    assert np.allclose(
        pes._closest_coordinates(point=(1,)), np.ones(shape=(1, 3)), atol=1e-10
    )


def test_distance():

    pes = TestPES(rs={(0, 1): np.array([1.0, 2.0])})

    assert np.isclose(pes._distance(point1=(0,), point2=(1,)), 1.0, atol=1e-10)

    pes = TestPES(
        rs={(0, 1): np.array([1.0, 2.0]), (1, 2): np.array([1.0, 2.0])}
    )

    assert np.isclose(
        pes._distance(point1=(0, 0), point2=(1, 1)), np.sqrt(2), atol=1e-10
    )


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
    assert np.isclose(consts[(0, 1)], 0.1, atol=1e-10)

    for i in range(3):
        consts = pes._constraints(point=(i,))
        assert np.isclose(consts[(0, 1)], 0.1 * (i + 1), atol=1e-10)


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

    pes = TestPES(rs={(0, 1): (1.0, 2.0, 3)})

    pes._energies = Energies(np.array([1.0, 0.01, 1.0]))
    assert len(list(pes._stationary_points())) == 1

    pes._energies = Energies(np.array([1.0, 1.1, 1.2]))
    assert len(list(pes._stationary_points())) == 0

    pes._energies = Energies(np.array([-1.0, -1.1, -1.2]))
    assert len(list(pes._stationary_points())) == 0

    # Idential energies will return the middle point
    pes._energies = Energies(np.array([-1.0, -1.0, -1.0]))
    stat_points = list(pes._stationary_points())
    assert len(stat_points) == 1
    assert stat_points[0] == (1,)


def test_stationary_points_2d():
    def energy(x, y):
        return 0.01 * (x * y - x**2 - x * y**2)

    pes = TestPES(rs={(0, 1): (-1.5, 1.5, 11), (1, 0): (-1.5, 1.5, 11)})

    pes._energies = Energies(energy(pes.r1, pes.r2))
    # pes.plot('tmp.pdf', interp_factor=0)
    # assert pes.shape == (50, 50)

    # Should have at least one stationary point. While in the
    # continuous surface there is 3, the finite surface may not have
    stat_points = list(pes._stationary_points())
    assert len(stat_points) > 0
    # The central point close to (0, 0) really should be present
    assert len([p for p in stat_points if p == (5, 5)]) == 1


def test_saddle_points_2d():
    def energy(x, y):
        return -(x**2) + y**2

    pes = TestPES(rs={(0, 1): (-1.0, 1.0, 11), (1, 0): (-1.0, 1.0, 11)})

    pes._energies = Energies(energy(pes.r1, pes.r2))
    # pes.plot('tmp.pdf', interp_factor=0)

    assert len(list(pes._stationary_points())) == 1

    # Should have at least one stationary point. While in the
    # continuous surface there is 3, the finite surface may not have
    points = list(pes._saddle_points())
    assert len(points) == 1

    p = points[0]

    # Saddle point should be close to (0, 0)
    assert np.isclose(pes.r1[p], 0.0, atol=0.1)
    assert np.isclose(pes.r2[p], 0.0, atol=0.1)


def test_numerical_gradient_harmonic_well():
    pes = harmonic_2d_pes()

    # Gradients should be initialised to nan
    assert all(np.isnan(g_k) for g_k in pes._gradients[1, 1])

    # With set gradients they should be a minimum at the centre
    # i.e close to r1=0, r2=0
    pes._set_gradients()

    # Norm is taken over the final axis (with length 2)
    norm_grad = np.linalg.norm(pes._gradients, axis=2)
    assert np.unravel_index(np.argmin(norm_grad), norm_grad.shape) == (10, 10)


def test_gradient_some_undefined_energies():

    pes = harmonic_2d_pes()
    i, j = pes.shape

    pes._energies[i // 3, j // 3] = np.nan

    # Should not raise any kind of exception, even though one of the
    # energies is undefined
    pes._set_gradients()

    # Should still have a stationary point, even if a point is undefined,
    # so long as it's not the stationary one
    assert len(list(pes._stationary_points())) > 0


def test_grad_neither_side_has_energy():

    pes = TestPES(rs={(0, 1): np.array([1.0, 2.0, 3.0])})

    mid_point = (1,)
    pes._energies[mid_point] = 1.0
    pes._set_gradients()

    # Cannot determine the numerical gradient if both enegies either side of
    # the mid-point do not have an energy
    assert np.all(np.isnan(pes._gradients[mid_point]))

    # Thus cannot be a minimum in |g|
    assert not pes._is_minimum_in_gradient(mid_point)
