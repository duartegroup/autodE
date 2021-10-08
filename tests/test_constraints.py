import pytest
from autode.constraints import Constraints


def test_base_properties():

    consts = Constraints()

    assert 'constraints' in str(consts).lower()
    assert 'constraints' in repr(consts).lower()

    assert not consts.any
    assert consts.distance is None
    assert consts.cartesian is None

    consts.update(distance={(0, 1): 1.0})
    assert consts.any
    assert consts.distance is not None and type(consts.distance) is dict
    assert consts.cartesian is None

    # Constraints should allow for setting
    consts.distance = {(0, 1): 2}
    assert int(consts.distance[(0, 1)]) == 2

    with pytest.raises(Exception):
        consts.distance = 1

    with pytest.raises(Exception):
        consts.distance = {(0, 1): -1.0}

    # Likewise with setting
    consts.cartesian = [0, 1]
    assert int(consts.cartesian[0]) == 0

    with pytest.raises(Exception):
        consts.cartesian = 5

    conts2 = Constraints(cartesian=[0], distance={(0, 1): 1.1})
    assert conts2.cartesian is not None and conts2.distance is not None

    assert '0' in str(conts2) and '1.1' in str(conts2)


def test_multiple_update():

    consts = Constraints()
    consts.update(distance={(0, 1): 1.0})
    consts.update(distance={(1, 0): 1.0})
    assert len(consts.distance) == 1

    # Non unique pairs should be skipped
    consts.update(distance={(0, 0): 1.0})
    assert len(consts.distance) == 1

    # Cannot have negative distances
    with pytest.raises(ValueError):
        consts.update(distance={(0, 1): -1.0})


def test_cartesian_update():

    conts = Constraints(cartesian=[0, 1])

    assert conts.any
    assert len(conts.cartesian) == 2

    # Should only have the unique components
    conts.update(cartesian=[0])
    assert len(conts.cartesian) == 2
