import pytest
from autode import Molecule
from autode.atoms import Atom
from autode.geom import are_coords_reasonable
from autode.smiles.parser import Parser, SMILESBonds
from autode.smiles.builder import Builder
from autode.exceptions import SMILESBuildFailed

parser = Parser()
builder = Builder()


def built_molecule_is_reasonable(smiles):
    """Is the molecule built from a SMILES string sensible?"""

    parser.parse(smiles)
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    mol.print_xyz_file(filename='tmp.xyz')

    return are_coords_reasonable(mol.coordinates)


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

    simple_smiles = ['C', 'CC', 'CCC', 'CCCC', 'CC(C)C']

    for smiles in simple_smiles:
        assert built_molecule_is_reasonable(smiles)


def test_long_alkane():
    """Should be able to build a long alkane without overlapping atoms"""

    assert built_molecule_is_reasonable(smiles='CCCCCCC')


def test_simple_multispecies():
    """Some simple molecules """

    assert built_molecule_is_reasonable(smiles='O')   # water
    assert built_molecule_is_reasonable(smiles='N')   # ammonia
    assert built_molecule_is_reasonable(smiles='B')   # BH3


def test_simple_multispecies2():
    """A small set of molecules with more than just carbon atoms"""

    assert built_molecule_is_reasonable(smiles='N#N')
    assert built_molecule_is_reasonable(smiles='OO')
    assert built_molecule_is_reasonable(smiles='O=[N]=O')
    assert built_molecule_is_reasonable(smiles='CN=C=O')


def test_simple_ring():
    """Small unsubstituted rings"""

    assert built_molecule_is_reasonable(smiles='C1CCCC1')  # cyclopentane
    assert built_molecule_is_reasonable(smiles='C1CCCCC1')  # cyclohexane
