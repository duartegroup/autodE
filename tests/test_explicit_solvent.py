import pytest
import os
import numpy as np
from scipy.spatial import distance_matrix
from autode.solvent import ExplicitSolvent
from autode.species.molecule import Molecule
from autode.atoms import Atom


def methane_mol():
    return Molecule(
        atoms=[
            Atom("C", 0.11105, -0.21307, 0.00000),
            Atom("H", 1.18105, -0.21307, 0.00000),
            Atom("H", -0.24562, -0.89375, 0.74456),
            Atom("H", -0.24562, -0.51754, -0.96176),
            Atom("H", -0.24562, 0.77207, 0.21720),
        ]
    )


def water_mol():
    return Molecule(
        atoms=[
            Atom("O", 1.64862, 0.46876, 0.00000),
            Atom("H", 2.61862, 0.46876, 0.00000),
            Atom("H", 1.32529, -0.28766, -0.51398),
        ]
    )


def test_explicit_solvent_gen():

    mol = Molecule(smiles="C", solvent_name="water")
    mol.explicitly_solvate(num=10)
    assert mol.solvent.is_explicit
    assert 75 < mol.solvent.dielectric < 80

    mol.print_xyz_file(filename="tmp.xyz")

    solv_mol = Molecule("tmp.xyz")
    assert solv_mol.n_atoms == (5 + 10 * 3)
    # Solute should be first in the file
    assert solv_mol.atoms[0].atomic_symbol == "C"

    coords = solv_mol.coordinates
    solute_coords = coords[:5]

    for i in range(10):
        solvent_mol_coords = coords[5 + i * 3 : 5 + (i + 1) * 3]
        assert np.min(distance_matrix(solute_coords, solvent_mol_coords)) > 1.9

    os.remove("tmp.xyz")

    mol.solvent = None
    mol.explicitly_solvate(num=10, solvent="water")
    assert mol.is_explicitly_solvated


def test_invalid_solvation():

    mol = Molecule(smiles="C", solvent_name="water")

    with pytest.raises(ValueError):
        mol.explicitly_solvate(num=-1)

    with pytest.raises(ValueError):
        mol.explicitly_solvate(num=0)

    solv = ExplicitSolvent(solute=mol, solvent=water_mol(), num=1)

    with pytest.raises(ValueError):
        solv.solvent_atom_idxs(-1)  # No solvent with index -1

    with pytest.raises(ValueError):
        solv.solvent_atom_idxs(1)  # or with index 1

    with pytest.raises(ValueError):
        mol.explicitly_solvate(1, solvent=-1)

    gas_phase_mol = Molecule(smiles="C")
    with pytest.raises(ValueError):
        gas_phase_mol.explicitly_solvate(num=1)


def test_too_close_to_solute():

    solute = methane_mol()
    water = water_mol()

    solv = ExplicitSolvent(solute=solute, solvent=water, num=1)
    assert solv._too_close_to_solute(
        water.coordinates, solute.coordinates, solute_radius=1.2
    )


def test_too_close_to_solvent():

    solv = ExplicitSolvent(solute=methane_mol(), solvent=water_mol(), num=2)
    assert solv.n_solvent_molecules == 2
    assert solv.solvent_n_atoms == 3

    # Solvent coordinates
    coords = np.array(
        [
            [1.64862, 0.46876, 0.00000],
            [2.61862, 0.46876, 0.00000],
            [1.32529, -0.28766, -0.51398],
            [1.69662, -0.04724, -1.83449],
            [2.66662, -0.04724, -1.83449],
            [1.37329, -0.88160, -1.46004],
        ]
    )

    second_solv_idxs = solv.solvent_atom_idxs(1)
    assert second_solv_idxs.tolist() == [3, 4, 5]

    assert solv._too_close_to_solvent(
        coords, solvent_idxs=second_solv_idxs, max_idx=1
    )


def test_equality():

    solv1 = ExplicitSolvent(solute=methane_mol(), solvent=water_mol(), num=2)

    solv2 = ExplicitSolvent(solute=methane_mol(), solvent=water_mol(), num=1)

    assert solv1 != 2
    assert solv1 != solv2
    assert solv1 == solv1


def test_solvate_with_molecule():

    solute = methane_mol()

    # Solvent should be able to be any valid solvent molecule
    solute.explicitly_solvate(solvent=methane_mol(), num=2)
    assert solute.n_atoms + solute.solvent.n_atoms == 15
