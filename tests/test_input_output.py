from autode.input_output import xyz_file_to_atoms, atoms_to_xyz_file
from autode.exceptions import XYZfileDidNotExist, XYZfileWrongFormat
from autode.atoms import Atom, Atoms
from . import testutils
import pytest
import os
here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'input_output.zip'))
def test_xyz_file_to_atoms():

    atoms = xyz_file_to_atoms(filename='opt_orca.xyz')
    assert len(atoms) == 5
    assert type(atoms) == Atoms
    assert type(atoms[0]) == Atom
    assert atoms[0].coord[0] == -0.137572

    with pytest.raises(XYZfileDidNotExist):
        xyz_file_to_atoms(filename='test')

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename='opt_orca_broken.xyz')

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename='opt_orca_broken2.xyz')

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename='wrong_ext.mol')


def test_xyz_file_incorrect_n_atoms():

    with open('test.xyz', 'w') as xyz_file:
        print('2',
              'wrongly declared number of atoms',
              'H   0.0   0.0   0.0',
              sep='\n', file=xyz_file)

    with pytest.raises(XYZfileWrongFormat):
        _ = xyz_file_to_atoms('test.xyz')

    os.remove('test.xyz')


def test_xyz_file_incorrect_first_line():
    with open('test.xyz', 'w') as xyz_file:
        print('XXX',
              'wrong first line',
              'H   0.0   0.0   0.0',
              sep='\n', file=xyz_file)

    with pytest.raises(XYZfileWrongFormat):
        _ = xyz_file_to_atoms('test.xyz')

    os.remove('test.xyz')


def test_making_xyz_file():

    atoms = [Atom('H'), Atom('H')]

    atoms_to_xyz_file(atoms, filename='test.xyz')
    atoms_to_xyz_file(atoms, filename='test.xyz', append=False)

    xyz_lines = open('test.xyz', 'r').readlines()
    assert len(xyz_lines) == 4

    # With append should add the next set of atoms to the same file
    atoms_to_xyz_file(atoms, filename='test.xyz', append=True)

    xyz_lines = open('test.xyz', 'r').readlines()
    assert len(xyz_lines) == 8

    with pytest.raises(AssertionError):
        # Requires some atoms
        atoms_to_xyz_file(atoms=None, filename='test.xyz')

        # Needs .xyz extension
        atoms_to_xyz_file(atoms, filename='test')

    os.remove('test.xyz')
