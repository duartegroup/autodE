from autode.complex import Complex
from autode.molecule import Molecule
from autode.atoms import Atom
import numpy as np
from copy import deepcopy
import pytest

h1 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
h2 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.0)

hydrogen = Molecule(name='H2', atoms=[h1, h2], charge=0, mult=1)
h = Molecule(name='H', atoms=[h1], charge=0, mult=2)

monomer = Complex(hydrogen)
dimer = Complex(hydrogen, hydrogen)
trimer = Complex(hydrogen, hydrogen, hydrogen)

h2_h = Complex(hydrogen, h)
h_h = Complex(h, h)


def test_complex_class():

    assert monomer.charge == 0
    assert monomer.mult == 1
    assert monomer.n_atoms == 2

    assert h2_h.charge == 0
    assert h2_h.mult == 2
    assert h2_h.n_atoms == 3

    assert h_h.mult == 3

    assert trimer.n_atoms == 6


def test_translation():

    # Monomer translation
    monomer_copy = deepcopy(monomer)
    monomer_copy.translate_mol(vec=np.array([1.0, 0.0, 0.0]), mol_index=0)

    assert np.linalg.norm(monomer_copy.atoms[0].coord - np.array([1.0, 0.0, 0.0])) < 1E-9
    assert np.linalg.norm(monomer_copy.atoms[1].coord - np.array([1.0, 0.0, 1.0])) < 1E-9

    # Dimer translation
    dimer_copy = deepcopy(dimer)
    dimer_copy.translate_mol(vec=np.array([1.0, 0.0, 0.0]), mol_index=1)

    assert np.linalg.norm(dimer_copy.atoms[2].coord - np.array([1.0, 0.0, 0.0])) < 1E-9
    assert np.linalg.norm(dimer_copy.atoms[3].coord - np.array([1.0, 0.0, 1.0])) < 1E-9

    # Cannot translate molecule index 2 in a complex with only 2 molecules
    with pytest.raises(AssertionError):
        dimer_copy.translate_mol(vec=np.array([1.0, 0.0, 0.0]), mol_index=2)


def test_rotation():

    dimer_copy = deepcopy(dimer)
    dimer_copy.rotate_mol(axis=np.array([1.0, 0.0, 0.0]), theta=np.pi,
                          origin=np.array([0.0, 0.0, 0.0]), mol_index=1)

    expected_coords = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0]])

    assert np.sum(expected_coords - dimer_copy.get_coordinates()) < 1E-9


def test_graph():

    hydrogen2 = deepcopy(hydrogen)
    hydrogen2.translate(vec=np.array([10, 0, 0]))

    dimer_shifted = Complex(hydrogen, hydrogen2)
    assert hasattr(dimer_shifted, 'graph')
    assert dimer_shifted.graph.number_of_edges() == 0
    assert dimer_shifted.graph.number_of_nodes() == 4
