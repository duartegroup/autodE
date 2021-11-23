import numpy as np
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

