import pytest
import numpy as np
from autode.units import ang
from autode.species import Molecule
from autode.atoms import Atom
from autode.pes.pes_nd import Distances1D, PESnD


def test_distances1d():

    rs = Distances1D([0.1, 0.2, -0.1], atom_idxs=(0, 1))

    assert np.isclose(rs.min, -0.1)
    assert np.isclose(rs.max, 0.2)
    assert 'dist' in repr(rs).lower()

    # Distances default to angstrom units
    assert rs.units == ang

    # Distances can be empty
    rs = Distances1D([], atom_idxs=(0, 1))
    assert len(rs) == 0

    # but cannot have negative atom indexes
    with pytest.raises(ValueError):
        Distances1D([], atom_idxs=(-1, 1))

    # or not have two atom indices
    with pytest.raises(ValueError):
        Distances1D([], atom_idxs=(0,))

    # or have non-integer types
    with pytest.raises(ValueError):
        Distances1D([], atom_idxs=(0.1, 1))


def test_pes_nd_attrs():

    pes = PESnD()

    # Empty PES has an empty tuple for a shape
    assert pes.shape == tuple()


def test_pes_nd_rs_init():

    # For a step-size of 0.1 Å there should be 10 steps in a single dimension
    pes = PESnD(rs={(0, 1): (1.0, 2.0, 0.1)})
    assert pes.shape == (10,)

    # Defining the number of steps should be equivalent
    pes = PESnD(rs={(0, 1): (1.0, 2.0, 10)})
    assert pes.shape == (10,)

    # while defining a a non-integer number raises a value error if no steps
    # are going to be performed
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1., 2., 10.1)})

    # or if only a single step is to be performed
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1., 2., 1)})


def test_pes_nd_rs_species_init():

    # Defining only the final distance and step size is
    # not supported without a species
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (2.0, 0.1)})

    # or the number of steps
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (2.0, 10)})

    # It is possible when a species is defined
    h2 = Molecule(atoms=[Atom('H'), Atom('H', x=1.0)])

    pes = PESnD(species=h2, rs={(0, 1): (1.5, 0.1)})
    assert pes.shape == (5,)

    pes = PESnD(species=h2, rs={(0, 1): (1.5, 5)})
    assert pes.shape == (5,)

    # negative increments give the same behaviour
    pes = PESnD(species=h2, rs={(0, 1): (0.5, -0.1)})
    assert pes.shape == (5,)

    # the atom indices must be in the molecule
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 2): (1.5, 5)})

    # and the number still be positive for negative steps
    assert PESnD(species=h2, rs={(0, 1): (0.5, 5)}).shape == (5,)
    assert PESnD(species=h2, rs={(0, 1): (0.5, 0.1)}).shape == (5,)


def test_pes_nd_rs_rounding():

    pes = PESnD(rs={(0, 1): (1.0, 2.03, 0.1)},
                allow_rounding=True)
    assert pes.shape == (10,)

    # If rounding is on then the step size should be rounded to support
    # an integer number of steps
    r1_arr = pes._rs[0]
    assert not np.isclose(r1_arr[0] - r1_arr[1], 0.1,
                          atol=1E-10)

    # but if rounding is not allowed then the final point should be
    # shifted
    pes = PESnD(rs={(0, 1): (1.0, 2.03, 0.1)},
                allow_rounding=False)

    # no rounding means there are 11 steps from 1.0 to 2.0 in 0.1 Å steps
    assert pes.shape == (11,)

    r1_arr = pes._rs[0]
    assert np.isclose(r1_arr[1] - r1_arr[0], 0.1, atol=1E-10)
    assert np.isclose(r1_arr[-1], 2.0, atol=1E-10)
