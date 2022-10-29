from autode.units import (
    ha,
    kjmol,
    kcalmol,
    ev,
    ang,
    a0,
    nm,
    pm,
    m,
    rad,
    deg,
    ha_per_ang,
    ev_per_ang,
    ha_per_a0,
)


def test_units():

    assert ha == "ha"
    assert ha == "hartree"

    # Ensure units have some base attributes
    for unit in (ha, kjmol, kcalmol, ev, ang, a0, nm, pm, m, rad, deg):

        assert unit.name != ""
        assert str(unit) != ""
        assert repr(unit) != ""
        assert len(unit.aliases) > 1
        assert unit.plot_name != ""


def test_composite_units():

    # More electron volts per angstrom than Hartees
    assert 1.0 * ha_per_ang.conversion < 1.0 * ev_per_ang.conversion

    # and fewer per bohr
    assert 1.0 * ha_per_ang.conversion > 1.0 * ha_per_a0.conversion
