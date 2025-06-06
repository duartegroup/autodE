import os

import numpy as np
import pytest

from autode.units import ang, ha, ha_per_ang, ha_per_a0, ev
from autode.values import (
    ValueArray,
    Gradient,
    Coordinate,
    Coordinates,
    MomentOfInertia,
    ForceConstant,
    _to,
)

here = os.path.dirname(os.path.abspath(__file__))


class TmpValues(ValueArray):
    implemented_units = [ha, ev]

    def __repr__(self):
        return ""


def test_base_arr():
    tmp_values = TmpValues(np.arange(2))
    assert tmp_values.units is None

    tmp_values = TmpValues(np.arange(2), units=ha)
    assert tmp_values.units == ha

    for item in (None, "a", 0, np.zeros(2)):
        # These are not the same! != calls __ne__
        assert not tmp_values == item
        assert tmp_values != item


def test_unit_retention():
    vals = TmpValues(np.array([0.1]), units=ev)
    assert vals.units == ev

    # Initialising an array from something with units should not default to the
    # default unit type (Hartrees for energies)
    vals1 = TmpValues(vals)
    assert vals1.units == ev


def test_coordinate():
    coord = Coordinate(0.0, 0.0, 0.0)
    assert coord.units == ang
    assert "coord" in repr(coord).lower()

    assert coord is not None
    assert np.allclose(coord, np.zeros(3))


def test_coordinates():
    arr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    coords = Coordinates(arr)
    assert coords.units == ang

    # Coordinates should reshape into n_atoms x 3
    coords = Coordinates(arr.flatten())
    assert coords.shape == (2, 3)

    assert "coord" in repr(coords).lower()


def test_moi():
    moi = MomentOfInertia(np.zeros(shape=(3, 3)))
    assert "i" in repr(moi).lower()


def test_gradients():
    # Default gradient units are Ha Å^-1
    gradients = Gradient(np.arange(3, dtype="f8"))
    assert gradients.units == ha_per_ang
    assert "grad" in repr(gradients).lower()

    gradients_ha_a0 = gradients.to(ha_per_a0)

    # Energy per bohr is smaller than per angstrom..
    assert all(
        g1 - g2 <= 0
        for g1, g2 in zip(gradients_ha_a0.flatten(), gradients.flatten())
    )


class Unit:
    conversion = 1.0
    aliases = ["unit"]

    def lower(self) -> str:
        return "unit"


class InvalidValue(float):
    implemented_units = [Unit]
    units = Unit()


def test_to_unsupported():
    with pytest.raises(ValueError):
        _ = _to(InvalidValue(), Unit(), inplace=True)


def test_inplace_modification():
    x = Gradient([[1.0, 1.0, 1.0]], units="Ha / Å")
    return_value = x.to_("eV / Å")
    assert return_value is None

    assert not np.allclose(x, np.ones(shape=(1, 3)))


def test_copy_conversion():
    x = Gradient([[1.0, 1.0, 1.0]], units="Ha / Å")
    y = x.to("eV / Å")
    assert not np.allclose(np.asarray(x), np.asarray(y))
    assert np.allclose(x, np.ones(shape=(1, 3)))


def test_force_constant():
    fc = ForceConstant(0.1)
    assert "force" in repr(fc).lower()

    # should be able to convert to Ha/a0^2 without any problems
    _ = fc.to("Ha/a0^2")


def test_pickle():
    """Regression test for https://github.com/duartegroup/autodE/issues/221"""
    import pickle

    x = Gradient([[1.0, 1.0, 1.0]], units=ha_per_ang)
    pickled_x = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
    unpickled_x = pickle.loads(pickled_x)
    assert unpickled_x.units == ha_per_ang
    assert unpickled_x == x


def test_load_old_mlptrain_npz():
    """Regression test for https://github.com/duartegroup/autodE/issues/372"""
    import numpy as np

    npz_file = os.path.join(here, "data", "old_mlptrain.npz")
    data = np.load(npz_file, allow_pickle=True)
    assert data["F_true"] is not None
