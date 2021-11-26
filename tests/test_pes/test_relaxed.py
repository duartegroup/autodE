import numpy as np
import pytest
from autode.pes.relaxed import RelaxedPESnD


def test_points_gen_idxs_1d():

    pes1d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})

    expected_points = [[(0,)], [(1,)], [(2,)]]
    for expected, true in zip(expected_points, pes1d._points_generator()):
        assert expected == true


def test_points_gen_idxs_2d():
    pes2d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2),
                             (1, 2): (1.0, 2.0, 2)})

    assert len(list(pes2d._points())) == 4

    # For a 2D grid there is points with indices that sum to 0, 1 and 2
    expected_points = [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]]

    for expected, true in zip(expected_points, pes2d._points_generator()):
        # order doesn't matter, so convert to sets
        assert set(expected) == set(true)


def test_points_gen_idxs_3d():
    pes3d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2),
                             (1, 2): (1.0, 2.0, 2),
                             (2, 3): (1.0, 2.0, 2)})

    assert pes3d.shape == (2, 2, 2)

    expected_points = [[(0, 0, 0)],
                       [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
                       [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
                       [(1, 1, 1)]]

    for expected, true in zip(expected_points, pes3d._points_generator()):
        # order doesn't matter, so convert to sets
        assert set(expected) == set(true)
