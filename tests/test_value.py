import pytest
import numpy as np
from autode.constants import Constants
from autode.units import (ha, kjmol, kcalmol, ev,
                          ang, a0, nm, pm, m,
                          rad, deg)
from autode.values import (Value, Distance, Angle,
                           Energy, PlottedEnergy, Energies)


class TmpValue(Value):
    def __str__(self):
        return ''


def test_base_value():

    val = TmpValue(0.0)
    assert val == 0.0
    assert hasattr(val, 'units')

    # Same representation as the string
    assert repr(val) == ""

    val2 = val.copy()
    val += 1

    assert val2 == 0.

    # Values are equal to default numpy isclose precision
    # (1e-08 as of 20/05/21)
    assert TmpValue(0.0) == TmpValue(1E-10)


def test_energy():

    e1 = Energy(0.0)
    assert 'energy' in str(e1).lower()
    assert type(e1.method_str) is str

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
    assert e1_kcal == 0.0

    # Conversion is not in place
    assert e1.units == ha

    e1 -= 0.1
    assert isinstance(e1, Energy)
    assert e1 == -0.1

    e1 *= 10
    assert np.isclose(e1, -1.0)

    # Should be able to add two energies
    e3 = e1 + Energy(1)
    assert e3 == 0.0

    e_kcal = Energy(1*Constants.ha_to_kcalmol, units=kcalmol)
    e_ha = Energy(1.0)

    # Values have implicit type conversion, left precedence
    assert np.isclose((e_ha + e_kcal), 2.0)
    assert (e_ha + e_kcal).units == ha

    # So the other way should add in kcal mol-1
    assert (e_kcal + e_ha) > 600

    # Comparisons are viable in different units
    assert Energy(1*Constants.ha_to_kcalmol, units=kcalmol) < Energy(2, units=ha)
    assert Energy(1*Constants.ha_to_kcalmol, units=kcalmol) > Energy(0.5, units=ha)

    # Math operations should not be in-place
    e1 = Energy(0.1)
    e2 = Energy(0.2)

    e3 = e1 + e2
    assert e3 == 0.3 and e1 == 0.1

    assert Energy(1.0) == Energy(1.0)
    assert Energy(1.0) != Energy(1.1)

    assert Energy(1.0, units=ha) == Energy(1.0*Constants.ha_to_kcalmol,
                                           units=kcalmol)

    assert np.isclose(Energy(1.0, units=kcalmol),
                      Energy(4.0, units=kjmol).to('kcal'),
                      atol=0.5)

    assert np.isclose(Energy(27, units=ev),
                      Energy(1, units=ha).to(ev),
                      atol=0.5)

    assert (Energy(1.0) * 10.0) == 10
    assert (10.0 * Energy(1.0)) == 10


def test_plotted_energy():

    # Equality does not check estimation
    assert PlottedEnergy(1, estimated=True) == PlottedEnergy(1, estimated=False)


def test_distance():

    assert 'dist' in str(Distance(1.0)).lower()

    # Bohrs are ~2 angstroms
    assert np.isclose(Distance(1.0, units=ang),
                      Distance(2.0, units=a0).to(ang),
                      atol=0.3)

    assert Distance(1.0, units=ang) == Distance(0.1, units=nm)
    assert Distance(1.0, units=ang) == Distance(100, units=pm)
    assert Distance(1.0, units=ang) == Distance(1E-10, units=m)


def test_angle():

    assert 'ang' in str(Angle(1.0)).lower()

    assert Angle(np.pi, units=rad) == Angle(180.0, units=deg)


def test_energies():

    energies = Energies()
    energies.append(Energy(1.0))
    energies.append(Energy(1.0))

    # Should not append identical energies
    assert len(energies) == 1
