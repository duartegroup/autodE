import numpy as np
from autode import geom
from autode.atoms import Atom


def test_are_coords_reasonable():

    good_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert geom.are_coords_reasonable(coords=good_coords) is True

    bad_coords1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    assert geom.are_coords_reasonable(coords=bad_coords1) is False

    bad_coords2 = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [2.0, 0.0, 0.0]])
    assert geom.are_coords_reasonable(coords=bad_coords2) is False


def test_length():
    assert geom.length(np.array([1.0, 1.0, 1.0])) == np.linalg.norm(np.array([1.0, 1.0, 1.0]))


def test_shifted_atoms():

    atoms = [Atom('H', 0.0, 0.0, 0.0), Atom('H', 0.0, 0.0, 2.0)]

    new_atoms = geom.get_shifted_atoms_linear_interp(atoms, bonds=[(0, 1)], final_distances=[1.0])

    # Linear interpolation of the coordinates should move the atom either end of the bond half way
    assert np.linalg.norm(new_atoms[0].coord - np.array([0.0, 0.0, 0.5])) < 1E-6
    assert np.linalg.norm(new_atoms[1].coord - np.array([0.0, 0.0, 1.5])) < 1E-6
