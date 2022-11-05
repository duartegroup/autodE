import pytest
import numpy as np
from autode.point_charges import PointCharge


def test_pc():

    pc = PointCharge(1.0)
    # Should initialise close to the origin
    assert np.linalg.norm(pc.coord) < 1e-6

    # and should be translatable
    pc.translate(1.0, 0.0, 0.0)
    assert np.linalg.norm(pc.coord - np.array([1.0, 0.0, 0.0])) < 1e-6

    # Should have the assigned charge (units of e)
    assert np.isclose(pc.charge, 1.0)

    # and be initialisable from a coord (for backwards compatibility)
    coord = np.array([-1.0, 0.0, 0.0])
    pc_from_coord = PointCharge(1.0, coord=coord)
    assert np.linalg.norm(pc_from_coord.coord - coord) < 1e-6


def test_pc_wrong_shape_coord():

    with pytest.raises(Exception):
        _ = PointCharge(charge=0, coord=np.zeros(4))
