import numpy as np
from autode import geom
from autode.atoms import Atom
import pytest


def test_are_coords_reasonable():

    good_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert geom.are_coords_reasonable(coords=good_coords)

    bad_coords1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    assert not geom.are_coords_reasonable(coords=bad_coords1)


def test_points_on_sphere():

    points = geom.get_points_on_sphere(n_points=4)

    # 4 points on a sphere equally spaced should be roughly √2 apart
    assert len(points) == 4
    assert np.abs(np.linalg.norm(points[0] - points[1]) - np.sqrt(2)) < 1e-6

    points = geom.get_points_on_sphere(n_points=2)
    # The algorithm isn't great at generated small numbers of points so 2 -> 3

    # 3 points on a sphere equally spaced should be roughly the diameter
    assert len(points) == 3
    assert np.abs(np.linalg.norm(points[0] - points[1]) - np.sqrt(3)) < 1e-6


def test_calc_rmsd():

    atoms = [
        Atom("C", 0.0009, 0.0041, -0.0202),
        Atom("H", -0.6577, -0.8481, -0.3214),
        Atom("H", -0.4585, 0.9752, -0.3061),
        Atom("H", 0.0853, -0.0253, 1.0804),
        Atom("H", 1.0300, -0.1058, -0.4327),
    ]

    atoms_rot = [
        Atom("C", -0.0009, -0.0041, -0.0202),
        Atom("H", 0.6577, 0.8481, -0.3214),
        Atom("H", 0.4585, -0.9752, -0.3061),
        Atom("H", -0.0853, 0.0253, 1.0804),
        Atom("H", -1.0300, 0.1058, -0.4327),
    ]

    coords1 = np.array([atom.coord for atom in atoms])
    coords2 = np.array([atom.coord for atom in atoms_rot])

    # Rotated coordinates should have almost 0 RMSD between them
    assert geom.calc_rmsd(coords1, coords2) < 1e-5

    # Coordinates need to have the same shape to calculate the RMSD
    with pytest.raises(AssertionError):
        _ = geom.calc_rmsd(coords1, coords2[1:])

    assert geom.calc_heavy_atom_rmsd(atoms, atoms_rot) < 1e-5

    # Permuting two hydrogens should generate a larger RMSD
    atoms_rot[2], atoms_rot[3] = atoms_rot[3], atoms_rot[2]
    rmsd = geom.calc_rmsd(
        coords1=np.array([atom.coord for atom in atoms]),
        coords2=np.array([atom.coord for atom in atoms_rot]),
    )

    assert rmsd > 0.1

    # While the heavy atom RMSD should remain unchanged
    assert geom.calc_heavy_atom_rmsd(atoms, atoms_rot) < 1e-6


def test_symm_matrix_from_ltril():

    m = geom.symm_matrix_from_ltril(array=[0, 1, 2])

    assert np.allclose(m, np.array([[0, 1], [1, 2]]))

    m = geom.symm_matrix_from_ltril(array=[0, -1, 4, 9, 0, 2])

    assert np.allclose(m, np.array([[0, -1, 9], [-1, 4, 0], [9, 0, 2]]))

    with pytest.raises(ValueError):
        _ = geom.symm_matrix_from_ltril(array=[1, 1])


def test_gram_schmidt():
    """Test the projection function by performing a single GS iteration
    see: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process"""

    u1 = np.random.uniform(-1, 1, size=3)

    v2 = np.random.uniform(-1, 1, size=3)
    u2 = v2 - geom.proj(u1, v2)

    # Resulting vectors should be orthogonal
    assert np.isclose(np.dot(u1, u2), 0.0)
