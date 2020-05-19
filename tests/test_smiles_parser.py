from autode.smiles_parser import SmilesParser
from autode.exceptions import InvalidSmilesString
from autode.atoms import Atom
from autode.geom import calc_rmsd
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


def test_stereochem():
    parser = SmilesParser()
    parser.parse_smiles('N[C@](Br)(O)C')
    parser_coords = [atom.coord for atom in parser.atoms][:5]
    desired_coords = [[1.26597, 0.60740, -0.09729],
                      [-0.26307, 0.59858, -0.07141],
                      [-0.91282, 2.25811, 0.01409],
                      [-0.72365, -0.12709, 1.01313],
                      [-0.64392, 0.13084, -1.00380]]
    assert calc_rmsd(parser_coords, desired_coords) < 0.5

    parser = SmilesParser()
    parser.parse_smiles('N[C@@H](Br)(O)')
    parser_coords = [atom.coord for atom in parser.atoms][:4] + [parser.atoms[6].coord]
    desired_coords = [[1.26597, 0.60740, -0.09729],
                      [-0.26307, 0.59858, -0.07141],
                      [-0.72365, -0.12709, 1.01313],
                      [-0.91282, 2.25811, 0.01409],
                      [-0.64392, 0.13084, -1.00380]]
    assert calc_rmsd(parser_coords, desired_coords) < 0.5

    parser = SmilesParser()
    parser.parse_smiles('F/C=C/F')
    parser_coords = [atom.coord for atom in parser.atoms]
    desired_coords = [[-4.14679, 1.36072, 0.92663],
                      [-3.58807, 1.44785, -0.00000],
                      [-2.26409, 1.31952, 0.00000],
                      [-1.70538, 1.40665, -0.92663],
                      [-4.11965, 1.64066, -0.92663],
                      [-1.73251, 1.12671, 0.92663]]
    assert calc_rmsd(parser_coords, desired_coords) < 0.5


def test_alt_ring_branch():

    parser = SmilesParser()
    parser.parse_smiles(smiles='[H][Rh]12([C]=O)([P+](C3=CC=CC4=C3OC5=C([P+](C6=CC=CC=C6)2C7=CC=CC=C7)'
                               'C=CC=C5C4(C)C)(C8=CC=CC=C8)C9=CC=CC=C9)CC1')

    # Should be valid and not raise InvalidSmilesString
    assert len(parser.atoms) == 84
