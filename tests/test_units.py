from autode.units import (ha, kjmol, kcalmol, ev,
                          ang, a0, nm, pm, m,
                          rad, deg)


def test_units():

    # Ensure units have some base attributes
    for unit in  (ha, kjmol, kcalmol, ev, ang, a0, nm, pm, m, rad, deg):

        assert unit.name != ''
        assert str(unit) != ''
        assert len(unit.aliases) > 1
        assert unit.plot_name != ''
