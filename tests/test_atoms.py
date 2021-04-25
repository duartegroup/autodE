from autode import atoms
from autode.atoms import Atom
import numpy as np
import pytest


def test_atoms():

    assert atoms.get_maximal_valance(atom_label='C') == 4
    assert atoms.get_maximal_valance(atom_label='Aa') == 6
    assert atoms.get_atomic_weight(atom_label='C') == 12.01
    assert atoms.get_atomic_weight(atom_label='Aa') == 70
    assert 0.9 < atoms.get_vdw_radius(atom_label='H') < 1.2
    assert 2 < atoms.get_vdw_radius(atom_label='Aa') < 3

    assert atoms.is_pi_atom(atom_label='C', valency=3) is True
    assert atoms.is_pi_atom(atom_label='C', valency=4) is False
    assert atoms.is_pi_atom(atom_label='Aa', valency=9) is False


def test_atom_collection():

    h2 = atoms.AtomCollection()
    assert h2.n_atoms == 0

    # Cannot set coordinates without atoms
    with pytest.raises(ValueError):
        h2.coordinates = np.array([1.0, 1.0, 1.0])

    h2.atoms = [Atom('H', 0.0, 0.0, 0.0), Atom('H')]
    assert h2.n_atoms == 2

    assert type(h2.coordinates) == np.ndarray

    coord = h2.coordinates[0]
    coord += 1.0

    # Shift of coordinates should not be in place
    assert not np.allclose(h2.coordinates[0], coord)

    # Cannot set coordinates with anything but a 3xn_atoms flat array, or
    # 2-dimensional array (matrix)
    with pytest.raises(AssertionError):
        h2.coordinates = np.array([])

    with pytest.raises(AssertionError):
        h2.coordinates = np.array([1.0, 0.1])

    with pytest.raises(AssertionError):
        h2.coordinates = np.array([[[1.0], [1.0]]])

    with pytest.raises(ValueError):
        h2.distance(-1, 0)

    with pytest.raises(ValueError):
        h2.distance(0, 2)


def test_atom():

    h = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
    assert h.label == 'H'
    assert h.atomic_number == 1
    assert h.atomic_symbol == 'H'
    assert not h.is_metal
    assert h.group == 1
    assert h.period == 1

    assert type(h.coord) == np.ndarray
    assert len(h.coord) == 3
    assert h.coord[0] == 0
    assert h.coord[1] == 0
    assert h.coord[2] == 0

    # Translate the H atom by 1 A in the z direction
    h.translate(vec=np.array([0.0, 0.0, 1.0]))
    assert np.linalg.norm(h.coord - np.array([0.0, 0.0, 1.0])) < 1E-6

    # Rotate the atom 180° (pi radians) in the x axis
    h.rotate(axis=np.array([1.0, 0.0, 0.0]), theta=np.pi)
    assert np.linalg.norm(h.coord - np.array([0.0, 0.0, -1.0])) < 1E-6

    # Perform a rotation about a different origin e.g. (1, 0, -1)
    h.rotate(axis=np.array([0.0, 0.0, 1.0]),
             theta=np.pi,
             origin=np.array([1.0, 0.0, -1.0]))
    assert np.linalg.norm(h.coord - np.array([2.0, 0.0, -1.0])) < 1E-6

    # Ensure that the atoms has a string representation
    assert len(str(h)) > 0

    assert Atom('C').atomic_number == 6
    assert Atom('C').period == 2
    assert Atom('C').group == 14

    dummy = atoms.DummyAtom(0.0, 0.0, 0.0)
    assert dummy.atomic_number == 0
    assert dummy.period == 0
    assert dummy.group == 0


def test_periodic_table():

    with pytest.raises(ValueError):
        _ = atoms.PeriodicTable.period(n=0)   # Periods are indexed from 1
        _ = atoms.PeriodicTable.period(n=8)   # and don't exceed 8

        _ = atoms.PeriodicTable.period(n=19)   # Groups don't exceed 18

    period2 = atoms.PeriodicTable.period(n=2)
    assert len(period2) == 8
    assert period2[0] == 'Li'

    assert len(atoms.PeriodicTable.period(n=1)) == 2
    assert len(atoms.PeriodicTable.period(n=3)) == 8
    assert len(atoms.PeriodicTable.period(n=4)) == 18

    group13 = atoms.PeriodicTable.group(n=13)
    assert 'B' in group13

    with pytest.raises(IndexError):
        _ = atoms.PeriodicTable.element(0, 0)
        _ = atoms.PeriodicTable.element(0, 3)

    assert atoms.PeriodicTable.element(2, 13) == 'B'
