import re
import pytest
from copy import deepcopy
from autode.atoms import metals
from autode.exceptions import InvalidSmilesString
from autode.smiles.base import SMILESStereoChem
from autode.smiles.parser import Parser


def test_base_properties():

    parser = Parser()

    assert parser.mult == 1
    assert parser.n_atoms == 0
    assert parser.charge == 0

    with pytest.raises(InvalidSmilesString):
        parser.smiles = "C*C"

    # Should allow for SMILES typos with leading or final empty spaces
    parser.parse(smiles="C ")

    # parser treats hydrogens as attributes of atoms
    assert parser.n_atoms == 1
    assert parser.atoms[0].n_hydrogens == 4

    assert str(parser.atoms[0]) is not None


def test_sq_brackets_parser():

    parser = Parser()
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles="[C")

    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles="[C[")

    # Needs at least one element
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles="[]")

    parser.parse(smiles="[C]")
    assert parser.n_atoms == 1
    assert parser.atoms[0].label == "C"
    assert parser.parsed  # should have parsed the SMILES fully

    parser.parse(smiles="[Cu]")
    assert parser.n_atoms == 1
    assert parser.atoms[0].label == "Cu"
    assert parser.atoms[0].charge == 0

    # Item in a square bracket must start with an element
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles="[674]")

    # Can't have multiple heavy (non-hydrogenic atoms) in a square bracket
    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles="[CC]")

    parser.parse(smiles="[CH3-]")
    assert parser.atoms[0].charge == -1
    assert parser.atoms[0].n_hydrogens == 3

    parser.parse(smiles="[Cu+]")
    assert parser.atoms[0].charge == 1

    parser.parse(smiles="[N+4]")
    assert parser.atoms[0].charge == 4
    assert parser.charge == 4

    parser.parse(smiles="[Cu++]")
    assert parser.atoms[0].charge == 2

    parser.parse(smiles="[N--]")
    assert parser.atoms[0].charge == -2

    parser.parse(smiles="[OH-1]")
    assert parser.atoms[0].charge == -1

    parser.parse(smiles="[NH-]")
    assert parser.atoms[0].charge == -1

    parser.parse(smiles="[N-2]")
    assert parser.atoms[0].charge == -2

    parser.parse(smiles="[Si@H3-]")
    assert parser.atoms[0].stereochem == SMILESStereoChem.TET_NORMAL
    assert parser.atoms[0].charge == -1
    assert parser.atoms[0].n_hydrogens == 3

    parser.parse(smiles="[C@@H2-]")
    assert parser.atoms[0].has_stereochem
    assert parser.atoms[0].stereochem == SMILESStereoChem.TET_INVERTED
    assert parser.atoms[0].charge == -1
    assert parser.atoms[0].n_hydrogens == 2
    assert parser.charge == -1


def test_multiple_atoms():

    parser = Parser()
    parser.parse(smiles="CC")
    assert parser.n_atoms == 2
    assert str(parser.bonds[0]) is not None
    assert parser.bonds[0].symbol == "-"

    assert all(atom.label == "C" for atom in parser.atoms)
    assert all(atom.charge == 0 for atom in parser.atoms)

    assert len(parser.bonds) == 1
    assert parser.bonds[0].order == 1

    parser.parse(smiles="[H][H]")
    assert parser.n_atoms == 2
    assert len(parser.bonds) == 1

    parser.parse(smiles="CN")
    assert parser.n_atoms == 2
    assert parser.bonds[0].order == 1

    parser.parse(smiles="N#N")
    assert parser.n_atoms == 2
    assert parser.bonds[0].order == 3

    parser.parse(smiles="C=O")
    assert parser.n_atoms == 2
    assert parser.bonds[0].order == 2

    parser.parse(smiles="CN=C=O")
    assert parser.n_atoms == 4
    assert len(parser.bonds) == 3


def test_branches():

    # Propane, but with a branch from the first atom
    parser = Parser()
    parser.parse(smiles="C(C)C")
    assert parser.n_atoms == 3
    assert parser.n_bonds == 2

    b1, b2 = parser.bonds
    assert (b1[0] == 0 and b1[1] == 1) or (b1[0] == 1 and b1[1] == 0)
    assert (b2[0] == 0 and b2[1] == 2) or (b2[0] == 2 and b2[1] == 0)

    # isobutane - properly branched
    parser.parse(smiles="CC(C)C")
    assert parser.n_atoms == 4
    assert parser.n_bonds == 3

    # octachlorodirhenate
    parser.parse(smiles="[Rh-](Cl)(Cl)(Cl)(Cl)$[Rh-](Cl)(Cl)(Cl)Cl")
    assert parser.n_atoms == 10
    assert parser.n_bonds == 9

    # should have a single quadruple bond
    assert any(bond.order == 4 for bond in parser.bonds)

    # 2-propyl-3-isopropyl-1-propanol
    parser.parse(smiles="OCC(CCC)C(C(C)C)CCC")
    assert parser.n_atoms == 13
    assert parser.n_bonds == 12

    # thiosulfate
    parser.parse(smiles="OS(=O)(=S)O")
    assert parser.n_atoms == 5
    assert parser.n_bonds == 4


def test_rings():

    parser = Parser()

    # cyclohexane
    parser.parse(smiles="C1CCCCC1")
    assert parser.n_atoms == parser.n_bonds == 6

    with pytest.raises(InvalidSmilesString):
        parser.parse(smiles="C1CCCCC")

    # Should be able to resolve multiple cyclohexenes to the same structure
    def n_double_bonds():
        return len([bond for bond in parser.bonds if bond.order == 2])

    cychexene_smiles = ["C=1CCCCC=1", "C=1CCCCC1", "C1CCCCC=1 "]
    for smiles in cychexene_smiles:

        parser.parse(smiles)
        assert parser.n_atoms == parser.n_bonds == 6
        assert n_double_bonds() == 1

    # perhydroisoquinoline
    parser.parse(smiles="N1CC2CCCC2CC1")
    assert parser.n_bonds == 10
    assert parser.n_atoms == 9

    # has 2 bonds that close rings, but plenty of bonds that form rings
    ring_bonds = [bond for bond in parser.bonds if bond.closes_ring]
    assert len(ring_bonds) == 2

    # Reusing ring closures is fine..
    bicylcohexyl_smiles = ["C1CCCCC1C2CCCCC2", "C1CCCCC1C1CCCCC1"]
    for smiles in bicylcohexyl_smiles:
        parser.parse(smiles)
        assert parser.n_atoms == 12
        assert parser.n_bonds == 13

    # Should be able to parse atoms with multiple dangling bonds to the
    # same atom
    parser.parse(smiles="C12(CCCCC1)CCCCC2")
    assert parser.n_atoms == 11

    # Should correct for atoms bonded to themselves
    parser.parse(smiles="C11")
    assert parser.n_atoms == 1
    assert parser.n_bonds == 0


def test_aromatic():

    parser = Parser()
    parser.parse(smiles="c1occc1")
    assert parser.n_atoms == 5
    assert parser.n_bonds == 5


def test_hydrogens():

    parser = Parser()

    # H atoms defined explicitly are treated as atoms
    parser.parse(smiles="[H]C([H])([H])[H]")
    assert parser.n_atoms == 5
    assert parser.n_bonds == 4

    assert len(parser.atoms) == 5


def test_cis_trans():

    parser = Parser()

    # Check that without defined stereochem the C-C double bond is present
    parser.parse(smiles="C(F)=CF")
    double_bond = next(bond for bond in parser.bonds if bond.order == 2)
    idx_i, idx_j = double_bond
    assert parser.atoms[idx_i].label == "C"
    assert parser.atoms[idx_j].label == "C"

    # trans (E) diflorouethene
    trans_dfe_smiles = ["F/C=C/F", r"F\C=C\F", r"C(\F)=C/F"]

    for smiles in trans_dfe_smiles:
        parser.parse(smiles)

        double_bond = next(bond for bond in parser.bonds if bond.order == 2)
        assert double_bond.is_trans(atoms=parser.atoms)
        assert not double_bond.is_cis(atoms=parser.atoms)

    # test the cis equivalent
    cis_dfe_smiles = [r"F\C=C/F", r"F\C=C/F", "C(/F)=C/F"]

    for smiles in cis_dfe_smiles:
        parser.parse(smiles)
        double_bond = next(bond for bond in parser.bonds if bond.order == 2)
        assert double_bond.is_cis(atoms=parser.atoms)

    parser.parse(smiles="F/C(CC)=C/F")
    double_bonds = [bond for bond in parser.bonds if bond.order == 2]
    assert len(double_bonds) == 1
    assert double_bonds[0].is_trans(atoms=parser.atoms)

    # Test allene stereochem
    parser.parse(smiles=r"F/C=C=C=C/F")
    # First carbon should be assigned stereochemistry
    assert parser.atoms[1].label == "C"
    assert parser.atoms[1].has_stereochem


def test_is_pi_atom():

    parser = Parser()

    parser.parse(smiles="C1=CC=CC=C1")  # benzene
    assert all(atom.is_pi for atom in parser.atoms)

    parser.parse(smiles="c1ccccc1")  # benzene, but with aromatic atoms
    assert all(atom.is_pi for atom in parser.atoms)


def test_implicit_hydrogens():

    parser = Parser()
    parser.parse(smiles="CC")
    # ethane carbons should have three hydrogens each
    assert parser.atoms[0].n_hydrogens == parser.atoms[1].n_hydrogens == 3

    parser.parse(smiles="B")
    assert parser.atoms[0].n_hydrogens == 3

    parser.parse(smiles="BC")
    assert parser.atoms[0].n_hydrogens == 2

    parser.parse(smiles="CBC")
    assert parser.atoms[1].n_hydrogens == 1

    parser.parse(smiles="P")
    assert parser.atoms[0].n_hydrogens == 3

    # For PF3 no hydrogens should be added
    parser.parse(smiles="FP(F)F")
    assert parser.atoms[1].n_hydrogens == 0

    # Should fill the valance of P up to 5 if currently is 4
    parser.parse(smiles="FP(F)(F)F")
    assert parser.bonds.n_involving(idx=1) == 4
    assert parser.atoms[1].n_hydrogens == 1

    # Should fill the valance of S up to 6 if currently is 5
    parser.parse(smiles="FS(F)(F)(F)F")
    assert parser.bonds.n_involving(idx=1) == 5
    assert parser.atoms[1].n_hydrogens == 1

    for halogen in ("F", "Cl", "Br", "I"):
        parser.parse(smiles=f"C{halogen}")
        assert parser.atoms[0].n_hydrogens == 3
        assert parser.atoms[1].n_hydrogens == 0

    # Should fill up to HCl etc.
    parser.parse(smiles="Cl")
    assert parser.n_atoms == 1
    assert parser.atoms[0].n_hydrogens == 1

    # Should not overfill an oxygen valance that is already exceeded
    parser.parse(smiles="CO(C)O")
    assert parser.atoms[1].n_hydrogens == 0

    parser.parse(smiles="O=[N]=O")
    assert parser.n_bonds == 2
    assert parser.bonds[0].order == parser.bonds[1].order == 2
    assert parser.atoms[1].n_hydrogens == 0

    # Should be able to parse aromatic structures
    parser.parse(smiles="c1ccccc1")
    assert all(atom.n_hydrogens == 1 for atom in parser.atoms)

    parser.parse(smiles="c1occc1")
    assert all(
        atom.n_hydrogens == 0 for atom in parser.atoms if atom.label == "O"
    )
    assert all(
        atom.n_hydrogens == 1 for atom in parser.atoms if atom.label == "C"
    )


def test_multiplicity():

    parser = Parser()

    # Test some simple examples
    parser.parse(smiles="[H]")
    assert parser.mult == 2

    parser.parse(smiles="C")
    assert parser.mult == 1

    # Multiple unpaired electrons default to singlets..
    parser.parse(smiles="C[C]C")
    assert parser.mult == 1


def test_double_bond_stereo_branch():

    parser = Parser()
    parser.parse(smiles=r"C/C([H])=C([H])/C")

    assert next(bond for bond in parser.bonds if bond.order == 2).is_trans(
        parser.atoms
    )


def test_alt_ring_branch():

    parser = Parser()
    smiles = (
        "O=C=[Rh]12(CC2)([H])=P(C3=CC=CC=C3)(C4=CC=CC=C4)C(C=CC=C5C6"
        "(C)C)=C5OC7=C6C=CC=C7P=1(C8=CC=CC=C8)C9=CC=CC=C9"
    )

    parser.parse(smiles)
    num_h_atoms = sum(atom.n_hydrogens for atom in parser.atoms)

    assert parser.n_atoms + num_h_atoms == 84


def test_ring_connectivity():

    parser = Parser()
    # Structure has a C-S(O2)-C motif
    parser.parse("CC12[C@@]3(CCC4)C4=C[C@@H](C1C=CO2)S(=O)3=O")

    atom_symbols_in_bonds = [
        {parser.atoms[i].label, parser.atoms[j].label} for i, j in parser.bonds
    ]

    n_c_s_bonds = len(
        [pair for pair in atom_symbols_in_bonds if pair == {"C", "S"}]
    )

    # and has two carbon-sulfur bonds
    assert n_c_s_bonds == 2


def test_multiplicity_metals():

    parser = Parser()

    parser.parse(smiles="[Na]C1=CC=CC=C1")
    assert parser.mult == 1


def test_aromatic_heteroatoms():

    parser = Parser()
    parser.parse(smiles="[nH]1cnnc1")

    # Should have 1 atom per carbon, plus one for the defined aromatic N
    assert sum(atom.n_hydrogens for atom in parser.atoms) == 3

    # also should not have any Hs for aromatic B
    parser.parse(smiles="c1c[cH-]bc1")
    assert sum(atom.n_hydrogens for atom in parser.atoms) == 4


def test_metal_in_smiles():
    def metal_in_smiles(smiles):
        at_strings = re.findall(r"\[.*?]", smiles)
        return any(
            metal in string for metal in metals for string in at_strings
        )

    assert not metal_in_smiles(smiles="CnnC")
    assert metal_in_smiles(smiles="CC[W]")
    assert metal_in_smiles(smiles="C[Pd]")
    assert metal_in_smiles(smiles="[Fe3+]CNO[W]")


def test_lots_of_smiles_rings():

    parser = Parser()

    # Should be able to parse a SMILES with ring closures with multiple
    # digits
    parser.parse(smiles="C%99CCCC%99")
    cyclopentane_atoms = deepcopy(parser.atoms)

    parser.parse(smiles="C1CCCC1")
    assert all(
        parser.atoms[i].label == cyclopentane_atoms[i].label
        for i in range(len(parser.atoms))
    )


def is_invalid(smiles):

    with pytest.raises(InvalidSmilesString):
        Parser().parse(smiles)


def is_valid(smiles):

    Parser().parse(smiles)  # Throws if invalid
    return True


def test_parse_ring_idx():

    # % ring closures must be followed by two numbers
    is_invalid("C%9CC")

    # and have at least two characters following the %
    is_invalid(smiles="C%")

    # and no non-integer characters
    is_invalid(smiles="C%$$")

    # Check that the function does reasonable things even if there is no
    # ring index present
    parser = Parser()

    parser._string = "CCCC"
    with pytest.raises(InvalidSmilesString):
        parser._parse_ring_idx(idx=0)


def test_parse_smiles_with_labels_no_h():

    parser = Parser()
    parser.parse("C[Br:777]")

    assert sum(["Br" == atom.atomic_symbol for atom in parser.atoms]) == 1

    br_atom = next(a for a in parser.atoms if a.label == "Br")
    assert br_atom.atom_class == 777


def test_parse_smiles_with_labels_with_h():

    parser = Parser()

    parser.parse("[CH4:2]")
    assert next(a for a in parser.atoms if a.label == "C").atom_class == 2


def test_parse_h3o_cation_smiles():

    assert is_valid("[O+H2]")


def test_parse_smiles_atom_class():

    assert is_valid("[H:1]")
    is_invalid("[H:1.1]")
    is_invalid("[H:a]")
