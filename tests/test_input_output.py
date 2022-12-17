import numpy as np

from autode.input_output import (
    xyz_file_to_atoms,
    atoms_to_xyz_file,
    xyz_file_to_molecules,
)
from autode.exceptions import XYZfileDidNotExist, XYZfileWrongFormat
from autode.atoms import Atom, Atoms
from autode.utils import work_in_tmp_dir
from . import testutils
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, "data", "input_output.zip"))
def test_xyz_file_to_atoms():

    atoms = xyz_file_to_atoms(filename="opt_orca.xyz")
    assert len(atoms) == 5
    assert type(atoms) == Atoms
    assert type(atoms[0]) == Atom
    assert atoms[0].coord[0] == -0.137572

    with pytest.raises(XYZfileDidNotExist):
        xyz_file_to_atoms(filename="test")

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename="opt_orca_broken.xyz")

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename="opt_orca_broken2.xyz")

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename="wrong_ext.mol")


@work_in_tmp_dir()
def test_xyz_file_incorrect_n_atoms():

    with open("test.xyz", "w") as xyz_file:
        print(
            "2",
            "wrongly declared number of atoms",
            "H   0.0   0.0   0.0",
            sep="\n",
            file=xyz_file,
        )

    with pytest.raises(XYZfileWrongFormat):
        _ = xyz_file_to_atoms("test.xyz")


@work_in_tmp_dir()
def test_xyz_file_incorrect_first_line():
    with open("test.xyz", "w") as xyz_file:
        print(
            "XXX",
            "wrong first line",
            "H   0.0   0.0   0.0",
            sep="\n",
            file=xyz_file,
        )

    with pytest.raises(XYZfileWrongFormat):
        _ = xyz_file_to_atoms("test.xyz")


@work_in_tmp_dir()
def test_making_xyz_file():

    atoms = [Atom("H"), Atom("H")]

    atoms_to_xyz_file(atoms, filename="test.xyz")
    atoms_to_xyz_file(atoms, filename="test.xyz", append=False)

    xyz_lines = open("test.xyz", "r").readlines()
    assert len(xyz_lines) == 4

    # With append should add the next set of atoms to the same file
    atoms_to_xyz_file(atoms, filename="test.xyz", append=True)

    xyz_lines = open("test.xyz", "r").readlines()
    assert len(xyz_lines) == 8

    with pytest.raises(AssertionError):
        # Requires some atoms
        atoms_to_xyz_file(atoms=None, filename="test.xyz")

    with pytest.raises(AssertionError):
        # Needs .xyz extension
        atoms_to_xyz_file(atoms, filename="test")


def _print_xyz_file_two_h_molecules(title_line: str = "title line") -> None:

    xyz_file_string = "1\n" f"{title_line}\n" "H 0.0 0.0 0.0"

    with open("tmp.xyz", "w") as file:
        print(xyz_file_string, xyz_file_string, sep="\n", file=file)


@work_in_tmp_dir()
def test_reading_multi_molecule_file_no_defined_values():

    _print_xyz_file_two_h_molecules()
    molecules = xyz_file_to_molecules("tmp.xyz")
    assert len(molecules) == 2
    m = molecules[0]

    # Molecules should be generated with sensible defaults
    assert m.energy is None
    assert m.charge == 0
    assert m.mult == 1


@work_in_tmp_dir()
def test_reading_multi_molecule_file_defined_charge_mult_energy():

    _print_xyz_file_two_h_molecules(title_line="E = -0.5 charge = 0 mult = 2")
    m = xyz_file_to_molecules("tmp.xyz")[0]
    assert m.energy is not None and np.isclose(m.energy, -0.5)
    assert m.charge == 0
    assert m.mult == 2
