from autode import atoms
from autode.atoms import Atom
import numpy as np


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


def test_atom():

    h = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
    assert h.label == 'H'
    assert h.atomic_number == 1
    assert h.atomic_symbol == 'H'
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
    h.rotate(axis=np.array([0.0, 0.0, 1.0]), theta=np.pi, origin=np.array([1.0, 0.0, -1.0]))
    assert np.linalg.norm(h.coord - np.array([2.0, 0.0, -1.0])) < 1E-6

    # Ensure that the atoms has a string representation
    assert len(str(h)) > 0

    assert Atom('C').atomic_number == 6
