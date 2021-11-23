import pytest
import numpy as np
from autode.units import ang
from autode.species import Molecule
from autode.values import Energy
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

    assert 'pes' in repr(pes).lower()
    assert 'pes' in repr(pes._energies).lower()


def test_pes_nd_rs_init():

    # For a step-size of 0.1 Å there should be 10 steps in a single dimension
    pes = PESnD(rs={(0, 1): (1.0, 2.0, 0.11)})
    assert pes.shape == (10,)

    # Defining the number of steps should be equivalent
    pes = PESnD(rs={(0, 1): (1.0, 2.0, 10)})
    assert pes.shape == (10,)

    # As is defining the array directly
    pes = PESnD(rs={(0, 1): np.linspace(1.0, 2.0, num=10)})
    assert pes.shape == (10,)

    # while defining a a non-integer number raises a value error if no steps
    # are going to be performed
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1., 2., 10.1)})

    # or if only a single step is to be performed
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1., 2., 1)})

    # or if there is only a float as the value
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): 1.0})

    # or if the tuple has <2 or >3 elements
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1.0,)})

    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1.0, 2.0, 10, 0.11
                               )})

    # or the final element in the tuple is not an int or float
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1.0, 'a')})

    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 1): (1.0, 2.0, 'a')})


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

    pes = PESnD(species=h2, rs={(0, 1): (1.4, 0.1)})
    assert pes.shape == (5,)

    pes = PESnD(species=h2, rs={(0, 1): (1.4, 5)})
    assert pes.shape == (5,)

    # negative increments give the same behaviour
    pes = PESnD(species=h2, rs={(0, 1): (0.6, -0.1)})
    assert pes.shape == (5,)

    # the atom indices must be in the molecule
    with pytest.raises(ValueError):
        _ = PESnD(rs={(0, 2): (1.4, 5)})

    # and the number still be positive for negative steps
    assert PESnD(species=h2, rs={(0, 1): (0.6, 5)}).shape == (5,)
    assert PESnD(species=h2, rs={(0, 1): (0.6, 0.1)}).shape == (5,)


def test_pes_nd_rs_rounding():

    pes = PESnD(rs={(0, 1): (1.0, 1.87, 0.1)},
                allow_rounding=True)
    assert pes.shape == (10,)

    # If rounding is on then the step size should be rounded to support
    # an integer number of steps
    r1_arr = pes._rs[0]
    assert not np.isclose(r1_arr[0] - r1_arr[1], 0.1,
                          atol=1E-10)

    # but if rounding is not allowed then the final point should be
    # shifted
    pes = PESnD(rs={(0, 1): (1.0, 1.87, 0.1)},
                allow_rounding=False)

    # no rounding means there are 11 steps from 1.0 to 2.0 in 0.1 Å steps
    assert pes.shape == (10,)

    r1_arr = pes._rs[0]
    # Final point should be rounded
    assert np.isclose(r1_arr[-1], 1.9, atol=1E-10)

    # and the step size fixed
    assert np.isclose(r1_arr[1] - r1_arr[0], 0.1, atol=1E-10)


def test_mesh():

    pes = PESnD(rs={(0, 1): (0.1, 0.3, 0.1),
                    (1, 2): (0.1, 0.3, 0.1)})
    assert pes.shape == (3, 3)

    assert hasattr(pes, 'r1')
    assert hasattr(pes, 'r2')

    assert np.allclose(pes.r1[0, 0], 0.1, atol=1E-10)
    assert np.allclose(pes.r2[0, 0], 0.1, atol=1E-10)

    # Second item in the matrix should modify r1
    # but leave unchanged r2 (row) i.e. the array be
    """
                r1
       ----------------------
       | (0, 0)   (0, 1)  ..
       |   .        .
    r2 |   .        . 
       |
    """
    assert np.allclose(pes.r1[0, 1], 0.2, atol=1E-10)
    assert np.allclose(pes.r2[0, 1], 0.1, atol=1E-10)

    # All elements on a non-calculated surface are initialised to 0
    for i in range(3):
        for j in range(3):
            assert np.isclose(pes[i, j], 0.0, atol=1E-10)

    assert isinstance(pes[0, 0], Energy)

