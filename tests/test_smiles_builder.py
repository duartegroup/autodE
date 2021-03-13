import pytest
from autode import Molecule
from autode.atoms import Atom
from autode.geom import are_coords_reasonable
from autode.smiles.parser import Parser, SMILESBonds
from autode.smiles.builder import Builder
from autode.exceptions import SMILESBuildFailed


def _test_base_builder():

    builder = Builder()

    # Builder needs SMILESAtom-s
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[Atom('H')], bonds=SMILESBonds())

    # Builder needs at least some atoms
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=None, bonds=[])

    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[], bonds=[])


def _test_explicit_hs():

    builder = Builder()

    parser = Parser()
    parser.parse(smiles='C')

    builder._set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

    # Should convert all implicit Hs to explicit atoms
    assert len(builder.atoms) == 5
    assert len(builder.bonds) == 4
    assert len([atom for atom in builder.atoms if atom.label == 'H']) == 4

    assert builder.graph.number_of_nodes() == 5
    assert builder.graph.number_of_edges() == 4

    parser.parse(smiles='CC(C)(C)C')
    assert len([True for (i, j) in parser.bonds
                if parser.atoms[i].label == parser.atoms[j].label == 'C']) == 4
    builder._set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)
    assert builder.n_atoms == 17


def test_simple_alkane():
    """A few simple linear and branched alkanes"""

    builder = Builder()
    parser = Parser()

    simple_smiles = ['C', 'CC', 'CCC', 'CCCC', 'CC(C)C']

    for smiles in simple_smiles:

        parser.parse(smiles)

        builder.build(parser.atoms, parser.bonds)
        mol = Molecule(atoms=builder.atoms)
        assert are_coords_reasonable(mol.coordinates)


def test_long_alkane():
    """Should be able to build a long alkane without overlapping atoms"""

    parser = Parser()
    parser.parse(smiles='CCCCCCC')

    builder = Builder()
    builder.build(parser.atoms, parser.bonds)

    mol = Molecule(atoms=builder.atoms)
    assert are_coords_reasonable(mol.coordinates)
