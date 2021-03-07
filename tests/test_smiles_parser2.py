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

    # Should allow for SMILES typos with leading or final empty spaces
    parser.parse(smiles='C ')
    assert parser.n_atoms == 1

    assert str(parser.atoms[0]) is not None


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
    assert str(parser.bonds[0]) is not None
    assert parser.bonds[0].symbol == '-'

    assert all(atom.label == 'C' for atom in parser.atoms)
    assert all(atom.charge == 0 for atom in parser.atoms)
    assert all(atom.n_hydrogens is None for atom in parser.atoms)

    assert len(parser.bonds) == 1
    assert parser.bonds[0].order == 1

    parser.parse(smiles='[H][H]')
    assert parser.n_atoms == 2
    assert len(parser.bonds) == 1

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
    assert parser.n_bonds == 2

    b1, b2 = parser.bonds
    assert (b1[0] == 0 and b1[1] == 1) or (b1[0] == 1 and b1[1] == 0)
    assert (b2[0] == 0 and b2[1] == 2) or (b2[0] == 2 and b2[1] == 0)

    # isobutane - properly branched
    parser.parse(smiles='CC(C)C')
    assert parser.n_atoms == 4
    assert parser.n_bonds == 3

    # octachlorodirhenate
    parser.parse(smiles='[Rh-](Cl)(Cl)(Cl)(Cl)$[Rh-](Cl)(Cl)(Cl)Cl')
    assert parser.n_atoms == 10
    assert parser.n_bonds == 9

    # should have a single quadruple bond
    assert any(bond.order == 4 for bond in parser.bonds)

    # 2-propyl-3-isopropyl-1-propanol
    parser.parse(smiles='OCC(CCC)C(C(C)C)CCC')
    assert parser.n_atoms == 13
    assert parser.n_bonds == 12

    # thiosulfate
    parser.parse(smiles='OS(=O)(=S)O')
    assert parser.n_atoms == 5
    assert parser.n_bonds == 4


def test_rings():

    parser = Parser()

    # cyclohexane
    parser.parse(smiles='C1CCCCC1')
    assert parser.n_atoms == parser.n_bonds == 6

    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles='C1CCCCC')

    # Should be able to resolve multiple cyclohexenes to the same structure
    def n_double_bonds():
        return len([bond for bond in parser.bonds if bond.order == 2])

    cychexene_smiles = ['C=1CCCCC=1', 'C=1CCCCC1', 'C1CCCCC=1 ']
    for smiles in cychexene_smiles:

        parser.parse(smiles)
        assert parser.n_atoms == parser.n_bonds == 6
        assert n_double_bonds() == 1

    # perhydroisoquinoline
    parser.parse(smiles='N1CC2CCCC2CC1')
    assert parser.n_bonds == 10
    assert parser.n_atoms == 9

    ring_bonds = [bond for bond in parser.bonds if bond.in_ring]
    assert len(ring_bonds) == 2

    # Reusing ring closures is fine..
    bicylcohexyl_smiles = ['C1CCCCC1C2CCCCC2', 'C1CCCCC1C1CCCCC1']
    for smiles in bicylcohexyl_smiles:
        parser.parse(smiles)
        assert parser.n_atoms == 12
        assert parser.n_bonds == 13

    # Should be able to parse atoms with multiple dangling bonds to the
    # same atom
    parser.parse(smiles='C12(CCCCC1)CCCCC2')
    assert parser.n_atoms == 11

    # Should correct for atoms bonded to themselves
    parser.parse(smiles='C11')
    assert parser.n_atoms == 1
    assert parser.n_bonds == 0


def test_aromatic():

    parser = Parser()
    parser.parse(smiles='c1occc1')
    assert parser.n_atoms == 5
    assert parser.n_bonds == 5


def test_hydrogens():

    parser = Parser()

    # H atoms defined explicitly are treated as atoms
    parser.parse(smiles='[H]C([H])([H])[H]')
    assert parser.n_atoms == 5
    assert parser.n_bonds == 4

    assert len(parser.canonical_atoms) == 5


def test_cis_trans():

    parser = Parser()

    # Check that without defined stereochem the C-C double bond is present
    parser.parse(smiles='C(F)=CF')
    double_bond = next(bond for bond in parser.bonds if bond.order == 2)
    idx_i, idx_j = double_bond
    assert parser.atoms[idx_i].label == 'C'
    assert parser.atoms[idx_j].label == 'C'

    # trans (E) diflorouethene
    trans_dfe_smiles = ['F/C=C/F', r'F\C=C\F', r'C(\F)=C/F']

    for smiles in trans_dfe_smiles:
        parser.parse(smiles)

        double_bond = next(bond for bond in parser.bonds if bond.order == 2)
        assert double_bond.is_trans(atoms=parser.atoms)
        assert not double_bond.is_cis(atoms=parser.atoms)

    # test the cis equivalent
    cis_dfe_smiles = [r'F\C=C/F', r'F\C=C/F', 'C(/F)=C/F']

    for smiles in cis_dfe_smiles:
        parser.parse(smiles)
        double_bond = next(bond for bond in parser.bonds if bond.order == 2)
        assert double_bond.is_cis(atoms=parser.atoms)

    parser.parse(smiles='F/C(CC)=C/F')
    double_bonds = [bond for bond in parser.bonds if bond.order == 2]
    assert len(double_bonds) == 1
    assert double_bonds[0].is_trans(atoms=parser.atoms)

    # Test allene stereochem
    parser.parse(smiles=r'F/C=C=C=C/F')
    # First carbon should be assigned stereochemistry
    assert parser.atoms[1].label == 'C'
    assert parser.atoms[1].has_stereochem


# TODO: (3) n_rad_electrons
