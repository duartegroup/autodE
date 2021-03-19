import pytest
import numpy as np
from autode import Molecule
from autode.atoms import Atom
from autode.geom import are_coords_reasonable
from autode.smiles.parser import Parser, SMILESBonds
from autode.smiles.builder import Builder, Dihedral
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


def test_dihedrals():

    trans = [Atom('C', -0.94807, -1.38247, -0.02522),
             Atom('C',  0.54343, -1.02958, -0.02291),
             Atom('C', -1.81126, -0.12418, -0.02130),
             Atom('C',  1.40662, -2.28788, -0.02401)]
    dihedral = Dihedral(idxs=[2, 0, 1, 3])
    assert np.isclose(dihedral.value(trans),
                      np.pi, atol=0.05)

    gauche = [Atom('C', 0.33245, -2.84500, 0.36258),
              Atom('C', 1.20438, -1.58016, 0.31797),
              Atom('C', 0.85514, -3.97306, -0.52713),
              Atom('C', 2.61201, -1.79454, 0.87465)]

    assert np.isclose(dihedral.value(gauche),
                      np.deg2rad(-64.5), atol=0.01)

    zero = [Atom('C', 0.0, 0.0, 0.0),
            Atom('C', 0.0, 0.0, 0.0),
            Atom('C', 0.0, 0.0, 0.0),
            Atom('C', 2.61201, -1.79454, 0.87465)]

    # Can't have a dihedral with vectors of zero length
    with pytest.raises(ValueError):
        _ = dihedral.value(zero)


def _test_simple_alkane():
    """A few simple linear and branched alkanes"""

    simple_smiles = ['C', 'CC', 'CCC', 'CCCC', 'CC(C)C']

    for smiles in simple_smiles:
        assert built_molecule_is_reasonable(smiles)


def _test_long_alkane():
    """Should be able to build a long alkane without overlapping atoms"""

    assert built_molecule_is_reasonable(smiles='CCCCCCC')


def _test_simple_multispecies():
    """Some simple molecules """

    assert built_molecule_is_reasonable(smiles='O')   # water
    assert built_molecule_is_reasonable(smiles='N')   # ammonia
    assert built_molecule_is_reasonable(smiles='B')   # BH3


def _test_simple_multispecies2():
    """A small set of molecules with more than just carbon atoms"""

    assert built_molecule_is_reasonable(smiles='N#N')
    assert built_molecule_is_reasonable(smiles='OO')
    assert built_molecule_is_reasonable(smiles='O=[N]=O')
    assert built_molecule_is_reasonable(smiles='CN=C=O')


def test_simple_ring():
    """Small unsubstituted rings"""

    parser.parse(smiles='C1CCCCC1')                          # cyclohexane
    builder._set_atoms_bonds(parser.atoms, parser.bonds)
    ring_dihedrals = list(builder._ring_dihedrals(ring_bond=[3, 4]))
    assert len(ring_dihedrals) == 3

    assert built_molecule_is_reasonable(smiles='C1CCCC1')     # cyclopentane
    assert built_molecule_is_reasonable(smiles='C1CCCCC1')    # cyclohexane
    assert built_molecule_is_reasonable(smiles='C1CCCCCC1')   # cycloheptane
    assert built_molecule_is_reasonable(smiles='C1CCCCCCC1')  # cycloctane
