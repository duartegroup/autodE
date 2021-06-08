import numpy as np
from autode.units import ang, ha, ha_per_ang, ha_per_a0
from autode.values import (ValueArray, Gradients, Coordinate)


class TmpValues(ValueArray):
    def __repr__(self):
        return ''


def test_base_arr():

    tmp_values = TmpValues(np.arange(2))
    assert tmp_values.units is None

    tmp_values = TmpValues(np.arange(2), units=ha)
    assert tmp_values.units == ha


def test_coordinate():
    coord = Coordinate(0.0, 0.0, 0.0)
    assert coord.units == ang

    assert coord is not None
    # Equality defaults to np.allclose
    assert coord == np.zeros(3)


def test_gradients():

    # Default gradient units are Ha Å^-1
    gradients = Gradients(np.arange(2, dtype='f8'))
    assert gradients.units == ha_per_ang
    assert 'grad' in repr(gradients).lower()

    gradients_ha_a0 = gradients.to(ha_per_a0)

    # Energy per bohr is smaller than per angstrom..
    assert all(g1 - g2 <= 0 for g1, g2 in zip(gradients_ha_a0, gradients))
