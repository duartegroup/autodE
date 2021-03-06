import pytest
from autode.exceptions import InvalidSmilesString
from autode.smiles.parser import Parser


def test_base_properties():
    parser = Parser()
    assert parser.n_rad_electrons == 0
    assert parser.n_atoms == 0

    # Cannot generate canonical atoms if the SMILES string has not been parsed
    with pytest.raises(ValueError):
        _ = parser.canonical_atoms

    with pytest.raises(InvalidSmilesString):
        parser.smiles = 'C*C'


def test_sq_brackets_parser():

    parser = Parser()
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles='[C')

    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles='[C[')

    # Needs at least one element
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles='[]')

    parser.parse(smiles='[C]')
    assert parser.n_atoms == 1
    assert parser.atoms[0].label == 'C'
    assert parser.parsed  # should have parsed the SMILES fully

    parser.parse(smiles='[Cu]')
    assert parser.n_atoms == 1
    assert parser.atoms[0].label == 'Cu'
    assert parser.atoms[0].charge == 0

    # Item in a square bracket must start with an element
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles='[674]')

    # Can't have multiple heavy (non-hydrogenic atoms) in a square bracket
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles='[CC]')

    parser.parse(smiles='[CH3-]')
    assert parser.atoms[0].charge == -1
    assert parser.atoms[0].n_hydrogens == 3

    parser.parse(smiles='[Cu+]')
    assert parser.atoms[0].charge == 1

    parser.parse(smiles='[N+4]')
    assert parser.atoms[0].charge == 4

    parser.parse(smiles='[Cu++]')
    assert parser.atoms[0].charge == 2

    parser.parse(smiles='[N--]')
    assert parser.atoms[0].charge == -2

    parser.parse(smiles='[OH-1]')
    assert parser.atoms[0].charge == -1

    parser.parse(smiles='[NH-]')
    assert parser.atoms[0].charge == -1

    parser.parse(smiles='[N-2]')
    assert parser.atoms[0].charge == -2

    parser.parse(smiles='[Si@H3-]')
    assert parser.atoms[0].stereochem == '@'
    assert parser.atoms[0].charge == -1
    assert parser.atoms[0].n_hydrogens == 3

    parser.parse(smiles='[C@@H2-]')
    assert parser.atoms[0].has_stereochem
    assert parser.atoms[0].stereochem == '@@'
    assert parser.atoms[0].charge == -1
    assert parser.atoms[0].n_hydrogens == 2


def test_multiple_atoms():

    parser = Parser()
    parser.parse(smiles='CC')
    assert parser.n_atoms == 2
    assert all(atom.label == 'C' for atom in parser.atoms)
    assert all(atom.charge == 0 for atom in parser.atoms)
    assert all(atom.n_hydrogens is None for atom in parser.atoms)

    assert len(parser.bonds) == 1
    assert parser.bonds[0].order == 1

    parser.parse(smiles='CN')
    assert parser.n_atoms == 2
    assert parser.bonds[0].order == 1

    parser.parse(smiles='N#N')
    assert parser.n_atoms == 2
    assert parser.bonds[0].order == 3

    parser.parse(smiles='C=O')
    assert parser.n_atoms == 2
    assert parser.bonds[0].order == 2

    parser.parse(smiles='CN=C=O')
    assert parser.n_atoms == 4
    assert len(parser.bonds) == 3


def test_branches():

    # Propane, but with a branch from the first atom
    parser = Parser()
    parser.parse(smiles='C(C)C')
    assert parser.n_atoms == 3
    assert len(parser.bonds) == 2

    b1, b2 = parser.bonds
    assert b1[0] == 0 and b1[1] == 1
    assert b2[0] == 0 and b2[1] == 2

    # isobutane - properly branched
    parser = Parser()
    parser.parse(smiles='CC(C)C')
    assert parser.n_atoms == 4
    assert len(parser.bonds) == 3
