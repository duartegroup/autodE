from autode.input_output import xyz_file_to_atoms
from autode.exceptions import XYZfileDidNotExist, XYZfileWrongFormat
from autode.atoms import Atom
import pytest
import os
here = os.path.dirname(os.path.abspath(__file__))


def test_xyz_file_to_atoms():

    os.chdir(os.path.join(here, 'data'))

    atoms = xyz_file_to_atoms(filename='opt_orca.xyz')
    assert len(atoms) == 5
    assert type(atoms) == list
    assert type(atoms[0]) == Atom
    assert atoms[0].coord[0] == -0.137572

    with pytest.raises(XYZfileDidNotExist):
        xyz_file_to_atoms(filename='test')

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename='opt_orca_broken.xyz')

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename='opt_orca_broken2.xyz')

    with pytest.raises(XYZfileWrongFormat):
        xyz_file_to_atoms(filename='opt_orca.out')

    os.chdir(here)
