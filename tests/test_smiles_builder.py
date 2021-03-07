import pytest
from autode.atoms import Atom
from autode.input_output import atoms_to_xyz_file
from autode.smiles.parser import Parser, SMILESBonds
from autode.smiles.builder import Builder
from autode.exceptions import SMILESBuildFailed


def test_base_builder():

    builder = Builder()

    # Builder needs SMILESAtom-s
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[Atom('H')], bonds=SMILESBonds())


def test_explicit_hs():

    builder = Builder()

    parser = Parser()
    parser.parse(smiles='C')

    builder._set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

    # Should convert all implicit Hs to explicit atoms
    assert len(builder.atoms) == 5
    assert len(builder.bonds) == 4
    assert len([atom for atom in builder.atoms if atom.label == 'H']) == 4


def test_simple_build():

    builder = Builder()

    parser = Parser()
    parser.parse(smiles='C')

    builder.build(parser.atoms, parser.bonds)
    atoms_to_xyz_file(atoms=builder.atoms, filename='tmp.xyz')
