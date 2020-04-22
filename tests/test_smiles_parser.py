from autode.smiles_parser import SmilesParser
from autode.exceptions import InvalidSmilesString
from autode.atoms import Atom
import pytest
import numpy as np


def test_parse_smiles():
    parser = SmilesParser()
    parser.parse_smiles('CC')

    assert len(parser.atoms) == 8
    assert len(parser.bonds) == 7
    assert parser.stereocentres == []
    assert parser.charge == 0
    assert parser.n_radical_electrons == 0

    with pytest.raises(InvalidSmilesString):
        parser.parse_smiles('CC1')


def test_divide_smiles():
    parser = SmilesParser()
    divided_smiles = list(parser.divide_smiles('CCl=[](())/C1=2'))
    assert divided_smiles == [('C', 'atom'), ('Cl', 'atom'), ('=', 'bond'), ('[]', 'bracket_atom'), ('(())', 'branch'), ('/', 'double_bond_stereochem'), ('C1=2', 'atom')]


def test_analyse_char():
    parser = SmilesParser()

    parser.analyse_char('C', 'atom')
    assert parser.atoms[0].label == 'C'
    assert len(parser.ring_dict) == 0

    parser.analyse_char('[C+]', 'bracket_atom')
    assert parser.atoms[1].label == 'C'
    assert parser.charge_dict[1] == 1

    parser.analyse_char('(C)', 'branch')
    assert parser.bonds[1] == (1, 2)
    assert parser.prev_atom_no == 1

    parser.analyse_char('=', 'bond')
    assert parser.bond_order_dict[2] == 2

    parser.analyse_char('/', 'double_bond_stereochem')
    assert parser.alkene_stereochem_dict[3] == '/'


def test_add_atom():
    parser = SmilesParser()
    parser.add_atom('C')
    assert parser.add_atom('Cl1') == '1'
    assert parser.bonds == [(0, 1)]
    assert parser.atoms[0].label == 'C'
    assert np.allclose(parser.atoms[0].coord, [0, 0, 0])
    assert parser.atoms[1].label == 'Cl'


def test_analyse_bond_ring_string():
    parser = SmilesParser()
    parser.analyse_bond_ring_string('=1')
    assert parser.ring_dict[1] == (0, 2)

    parser.prev_atom_no = 2
    parser.analyse_bond_ring_string('1')
    assert parser.bonds == [(0, 2)]
    assert parser.ring_dict == {}

    parser.analyse_bond_ring_string('23')
    assert parser.ring_dict[2] == (2, None)
    assert parser.ring_dict[3] == (2, None)

    parser.prev_atom_no = 5
    parser.analyse_bond_ring_string('%15')
    assert parser.ring_dict[15] == (5, None)


def test_analyse_atom_details():
    parser = SmilesParser()
    parser.add_atom('C')
    parser.analyse_atom_details('H+2@')
    assert parser.charge_dict[0] == 2
    assert parser.hydrogen_dict[0] == 1
    assert parser.stereochem_dict[0] == '@td'

    parser.add_atom('C')
    parser.analyse_atom_details('-2@@')
    assert parser.charge_dict[1] == -2
    assert parser.hydrogen_dict[1] == 0
    assert parser.stereochem_dict[1] == '@@td'


def test_add_hs():
    parser = SmilesParser()
    parser.add_atom('C')
    parser.analyse_atom_details('H2')
    parser.add_atom('C')
    parser.add_atom('C')
    parser.bond_order_dict[1] = 2
    parser.add_hs()
    assert len(parser.atoms) == 8
    assert len(parser.bonds) == 7
    assert parser.n_radical_electrons == 1


# def test_stereochem():
#     pass
