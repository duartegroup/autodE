import pytest
import numpy as np
from autode.constants import Constants
from autode.units import (ha, kjmol, kcalmol, ev,
                          ang, a0, nm, pm, m,
                          rad, deg)
from autode.values import (Value, Distance, Angle,
                           Energy, PlottedEnergy, Energies,
                           PotentialEnergy, Enthalpy, FreeEnergy,
                           FreeEnergyCont, EnthalpyCont,
                           Frequency)


class TmpValue(Value):
    def __repr__(self):
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

    with pytest.raises(ValueError):
        Energy(0.0, units='not_an_energy_unit')

    e1 = Energy(0.0)
    assert 'energy' in repr(e1).lower()
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


def test_enthalpy():

    assert Enthalpy(1.0) != PotentialEnergy(1.0)
    assert PotentialEnergy(1.0) != Enthalpy(1.0)


def test_free_energy():

    assert FreeEnergy(1.0) != PotentialEnergy(1.0)
    assert FreeEnergy(1.0) != Enthalpy(1.0)
    assert PotentialEnergy(1.0) != FreeEnergy(1.0)


def test_plotted_energy():

    # Equality does not check estimation
    assert PlottedEnergy(1, estimated=True) == PlottedEnergy(1, estimated=False)


def test_distance():

    assert 'dist' in repr(Distance(1.0)).lower()

    # Bohrs are ~2 angstroms
    assert np.isclose(Distance(1.0, units=ang),
                      Distance(2.0, units=a0).to(ang),
                      atol=0.3)

    assert Distance(1.0, units=ang) == Distance(0.1, units=nm)
    assert Distance(1.0, units=ang) == Distance(100, units=pm)
    assert Distance(1.0, units=ang) == Distance(1E-10, units=m)


def test_angle():

    assert 'ang' in repr(Angle(1.0)).lower()

    assert Angle(np.pi, units=rad) == Angle(180.0, units=deg)


def test_energies():

    energies = Energies()
    energies.append(Energy(1.0))
    energies.append(Energy(1.0))

    # Should not append identical energies
    assert len(energies) == 1

    energies = Energies(Energy(1.0), FreeEnergy(0.1))

    assert energies.last(FreeEnergy) == FreeEnergy(0.1)

    assert 'free' in repr(FreeEnergy(0.0)).lower()
    assert 'enthalpy' in repr(Enthalpy(0.0)).lower()

    assert 'cont' in repr(FreeEnergyCont(0.0)).lower()
    assert 'cont' in repr(EnthalpyCont(0.0)).lower()

    # Check that adding an energy that is already present moves it to the end
    energies = Energies()
    energies.append(Energy(1.0))
    energies.append(Energy(5.0))
    energies.append(Energy(1.0))

    assert energies.last(Energy) == 1.0


def test_freqs():

    # Negative frequencies are actually imaginary (accepted convention in QM
    # codes)
    assert Frequency(-1.0).is_imaginary
    assert not Frequency(1.0).is_imaginary

    assert Frequency(-1.0) != Frequency(1.0)
    assert Frequency(-1.0).real == Frequency(1.0)
