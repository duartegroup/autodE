import numpy as np
import pytest
from autode import atoms
from autode.atoms import Atom, Atoms
from autode.values import Angle, Coordinate


def test_functions():

    assert atoms.get_maximal_valance(atom_label='C') == 4
    assert atoms.get_maximal_valance(atom_label='Aa') == 6
    assert 11.9 < atoms.get_atomic_weight(atom_label='C') < 12.1
    assert atoms.get_atomic_weight(atom_label='Aa') == 70
    assert 0.9 < atoms.get_vdw_radius(atom_label='H') < 1.2
    assert 2 < atoms.get_vdw_radius(atom_label='Aa') < 3

    assert atoms.is_pi_atom(atom_label='C', valency=3) is True
    assert atoms.is_pi_atom(atom_label='C', valency=4) is False
    assert atoms.is_pi_atom(atom_label='Aa', valency=9) is False


def test_atoms():

    empty_atoms = Atoms()

    # Undefined COM with no atoms
    with pytest.raises(ValueError):
        _ = empty_atoms.com

    h_atoms = Atoms([Atom('H'), Atom('H', x=1.0)])
    assert isinstance(h_atoms.com, Coordinate)

    assert np.allclose(h_atoms.com, np.array([0.5, 0.0, 0.0]))

    # COM is weighted by mass, so the x-coordinate
    ch_atoms = Atoms([Atom('H'), Atom('C', x=1.0)])

    assert 0.5 < ch_atoms.com.x < 1.0
    assert ch_atoms.com.y == 0.0
    assert ch_atoms.com.z == 0.0


def test_atom_collection_base():

    h2 = atoms.AtomCollection()
    assert h2.n_atoms == 0
    assert h2.coordinates is None

    # Cannot set coordinates without atoms
    with pytest.raises(ValueError):
        h2.coordinates = np.array([1.0, 1.0, 1.0])

    h2.atoms = [Atom('H', 0.0, 0.0, 0.0), Atom('H')]
    assert h2.n_atoms == 2

    # Should be able to set coordinate from a flat array (row major)
    h2.coordinates = np.zeros(shape=(6,))
    assert h2.coordinates[0] is not None
    assert h2.n_atoms == 2

    assert np.isclose(h2.distance(0, 1), 0.0, atol=1E-5)

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


def test_atom_collection_angles():

    h2o = atoms.AtomCollection()
    h2o.atoms = [Atom('H', x=-1.0),
                 Atom('O'),
                 Atom('H', x=1.0)]

    # Should default to more human readable degree units
    assert np.isclose(h2o.angle(0, 1, 2).to('deg'), 180)
    assert np.isclose(h2o.angle(0, 1, 2).to('degrees'), 180)

    # No -1 atom
    with pytest.raises(ValueError):
        _ = h2o.angle(-1, 0, 1)

    # Angle is not defined when one vector is the zero vector
    with pytest.raises(ValueError):
        _ = h2o.angle(0, 0, 1)

    # Angles default to radians
    assert np.isclose(np.abs(h2o.angle(0, 1, 2)), np.pi)

    with pytest.raises(TypeError):
        _ = h2o.angle(0, 1, 2).to('not a unit')

    assert isinstance(h2o.angle(0, 1, 2).copy(), Angle)

    h2o.atoms[1].coord = np.array([-0.8239, -0.5450, 0.0000])
    h2o.atoms[2].coord = np.array([0.8272, -0.5443, 0.0000])

    assert 90 < h2o.angle(0, 1, 2).to('deg') < 120


def test_atom_collection_dihedral():

    h2o2 = atoms.AtomCollection()
    h2o2.atoms = [Atom('O', -0.85156, -0.20464,  0.31961),
                  Atom('O',  0.41972,  0.06319,  0.10395),
                  Atom('H', -1.31500,  0.08239, -0.50846),
                  Atom('H',  0.58605,  0.91107,  0.59006)]

    assert np.isclose(h2o2.dihedral(2, 0, 1, 3).to('deg'),
                      100.8,
                      atol=1.0)

    # Undefined dihedral with a zero vector between teo atoms
    with pytest.raises(ValueError):
        h2o2.atoms[0].coord = np.zeros(3)
        h2o2.atoms[1].coord = np.zeros(3)

        _ = h2o2.dihedral(2, 0, 1, 3)


def test_atom():

    h = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
    assert h.label == 'H'
    assert h.atomic_number == 1
    assert h.atomic_symbol == 'H'
    assert not h.is_metal
    assert h.tm_row is None
    assert h.group == 1
    assert h.period == 1

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

    fe = Atom(atomic_symbol='Fe')
    assert fe.tm_row == 1


def test_atom_coord_setting():

    atom = Atom('H', 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        atom.coord = None

    with pytest.raises(ValueError):
        atom.coord = [1.0, 10]

    with pytest.raises(ValueError):
        atom.coord = 1.0, 1.0

    atom.coord = np.array([1.0, 0.0, 0.0])
    assert np.allclose(atom.coord.to('nm'), np.array([0.1, 0.0, 0.0]))


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

    with pytest.raises(Exception):
        _ = atoms.PeriodicTable.group(0)   # No group 0
        _ = atoms.PeriodicTable.group(19)  # or 19

    with pytest.raises(IndexError):
        _ = atoms.PeriodicTable.element(0, 0)
        _ = atoms.PeriodicTable.element(0, 3)

    with pytest.raises(Exception):
        _ = atoms.PeriodicTable.transition_metals(row=0)
        _ = atoms.PeriodicTable.transition_metals(row=10)

    assert 'Fe' in atoms.PeriodicTable.transition_metals(row=1)

    assert atoms.PeriodicTable.element(2, 13) == 'B'
