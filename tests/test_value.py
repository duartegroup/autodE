import pytest
import numpy as np
from autode.units import ha, kcalmol, deg
from autode.values import Value, Energy


class TmpValue(Value):
    def __str__(self):
        return ''


def test_base_value():

    val = TmpValue(0.0)
    assert val == 0.0
    assert hasattr(val, 'units')


def test_energy():

    e1 = Energy(0.0)
    assert 'energy' in str(e1).lower()
    assert e1 == 0.0
    assert -0.001 < e1 < 0.001
    assert -0.001 <= e1 <= 0.001

    assert e1.units == ha

    # Cannot convert to a non-existent unit
    with pytest.raises(TypeError):
        _ = e1.to('xxx')

    # or to a non-energy unit
    with pytest.raises(TypeError):
        _ = e1.to(deg)

    # but can convert to a different type of energy unit
    e1_kcal = e1.to(kcalmol)
    assert e1_kcal.x == 0.0

    # Conversion is not in place
    assert e1.units == ha

    e1 -= 0.1
    assert isinstance(e1, Energy)
    assert e1.x == -0.1

    e1 *= 10
    assert np.isclose(e1.x, -1.0)

    # Should be able to add two energies
    e3 = e1 + Energy(1)
    assert e3.x == 0.0

    e_kcal = Energy(627.509, units=kcalmol)
    e_ha = Energy(1.0)

    # Values have implicit type conversion, left precedence
    assert np.isclose((e_ha + e_kcal).x, 2.0)
    assert (e_ha + e_kcal).units == ha

    # So the other way should add in kcal mol-1
    assert (e_kcal + e_ha).x > 600

    # Comparisons are viable in different units
    assert Energy(627.509, units=kcalmol) < Energy(2, units=ha)
    assert Energy(627.509, units=kcalmol) > Energy(0.5, units=ha)

    # Math operations should not be in-place
    e1 = Energy(0.1)
    e2 = Energy(0.2)

    e3 = e1 + e2
    assert e3 == 0.3 and e1 == 0.1
