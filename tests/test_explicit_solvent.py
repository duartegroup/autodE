import pytest
from autode.species.molecule import Molecule


def test_explicit_solvent_gen():

    mol = Molecule(smiles='C', solvent_name='water')
    mol.explicitly_solvate(num=100)
    mol.print_xyz_file(filename='tmp.xyz')


if __name__ == '__main__':

    test_explicit_solvent_gen()
