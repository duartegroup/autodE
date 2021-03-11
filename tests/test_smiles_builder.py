import pytest
from autode import Molecule
from autode.atoms import Atom
from autode.geom import are_coords_reasonable
from autode.smiles.parser import Parser, SMILESBonds
from autode.smiles.builder import Builder
from autode.exceptions import SMILESBuildFailed


def test_base_builder():

    builder = Builder()

    # Builder needs SMILESAtom-s
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[Atom('H')], bonds=SMILESBonds())

    # Builder needs at least some atoms
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=None, bonds=[])

    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[], bonds=[])


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

    simple_smiles = ['C', 'CC', 'CCC']

    for smiles in simple_smiles:

        parser.parse(smiles)

        builder.build(parser.atoms, parser.bonds)
        mol = Molecule(atoms=builder.atoms)
        mol.print_xyz_file(filename='tmp.xyz')

        assert are_coords_reasonable(mol.coordinates)
