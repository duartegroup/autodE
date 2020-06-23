from autode import bond_rearrangement as br
from autode.species.molecule import Molecule
from autode.bond_rearrangement import BondRearrangement
from autode.species.complex import ReactantComplex, ProductComplex
from autode.atoms import Atom
from autode.mol_graphs import is_isomorphic
from autode.mol_graphs import make_graph
import networkx as nx
import os


# Some of the 'reactions' here are not physical, hence for some the graph will be regenerated allowing for
# invalid hydrogen valencies


def test_multiple_possibilities():
    r1 = Molecule(name='h_dot', smiles='[H]')
    r2 = Molecule(name='methane', smiles='C')
    p1 = Molecule(name='h2', smiles='[HH]')
    p2 = Molecule(name='ch3_dot', smiles='[CH3]')

    reac = ReactantComplex(r1, r2)
    reac.print_xyz_file()

    rearrs = br.get_bond_rearrangs(reac, ProductComplex(p1, p2),
                                   name='H_subst')
    os.remove('H_subst_bond_rearrangs.txt')

    # All H abstractions are the same
    assert len(rearrs) == 1


def test_bondrearr_obj():
    # Reaction H + H2 -> H2 + H
    rearrang = br.BondRearrangement(forming_bonds=[(0, 1)],
                                    breaking_bonds=[(1, 2)])

    assert rearrang.n_fbonds == 1
    assert rearrang.n_bbonds == 1
    assert rearrang.__str__() == '0-1_1-2'

    rearrag2 = br.BondRearrangement(forming_bonds=[(0, 1)],
                                    breaking_bonds=[(1, 2)])
    assert rearrag2 == rearrang

    h2_h_mol = Molecule(name='mol', atoms=[Atom('H', 0.0, 0.0, 0.0),
                                           Atom('H', 0.0, 0.0, -1.0),
                                           Atom('H', 0.0, 0.0, 1.0)])

    active_atom_nl = rearrang.get_active_atom_neighbour_lists(mol=ReactantComplex(h2_h_mol), depth=1)
    assert len(active_atom_nl) == 3
    assert active_atom_nl == [['H'], ['H'], ['H']]


def test_get_bond_rearrangs():

    if os.path.exists('test_bond_rearrangs.txt'):
        os.remove('test_bond_rearrangs.txt')

    # ethane --> Ch3 + Ch3
    reac = Molecule(smiles='CC')
    prod = Molecule(atoms=[Atom('C', -8.3, 1.4, 0.0),
                           Atom('C', 12, 1.7, -0.0),
                           Atom('H', -8.6, 0.5, -0.5),
                           Atom('H', -8.6, 2.3, -0.4),
                           Atom('H', -8.6, 1.3, 1),
                           Atom('H', 12.3, 1.7, -1.0),
                           Atom('H', 12.4, 0.8, 0.4),
                           Atom('H', 12.3, 2.5, 0.5)])

    assert br.get_bond_rearrangs(ReactantComplex(reac), ProductComplex(prod), name='test') == [br.BondRearrangement(breaking_bonds=[(0, 1)])]

    # Rerunning the get function should read test_bond_rearrangs.txt, so modify it, swapping 0 and 1 in the breaking
    # bond then reopen
    with open('test_bond_rearrangs.txt', 'w') as rearr_file:
        print('fbond\n'
              'bbonds\n'
              '1 0\n'
              'end', file=rearr_file)

    rearr = br.get_bond_rearrangs(ReactantComplex(reac), ProductComplex(prod), name='test')[0]
    assert rearr == BondRearrangement(breaking_bonds=[(1, 0)])

    assert br.get_bond_rearrangs(ReactantComplex(prod), ProductComplex(reac), name='test2') is None

    # If reactants and products are identical then the rearrangement is undetermined
    assert br.get_bond_rearrangs(ReactantComplex(reac), ProductComplex(reac), name='test3') is None

    os.remove('test_bond_rearrangs.txt')


def test_two_possibles():

    ch2ch3f = Molecule(name='radical', charge=0, mult=2,
                       smiles='FC[C]([H])[H]')

    ch3ch2f = Molecule(name='radical', charge=0, mult=2,
                       smiles='C[C]([H])F')

    rearrs = br.get_bond_rearrangs(ReactantComplex(ch2ch3f), ProductComplex(ch3ch2f),
                                   name='H_migration')

    # There are two possibilities for H migration by they should be considered the same
    assert len(rearrs) == 1
    os.remove('H_migration_bond_rearrangs.txt')
    
    
def test_add_bond_rearrang():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0)])
    assert br.add_bond_rearrangment([], reac, prod, [], [(0, 1)]) == [br.BondRearrangement(breaking_bonds=[(0, 1)])]


def test_generate_rearranged_graph():
    init_graph = nx.Graph()
    final_graph = nx.Graph()
    init_edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    final_edges = [(0, 1), (2, 3), (3, 4), (4, 5), (5, 6)]
    for edge in init_edges:
        init_graph.add_edge(*edge)
    for edge in final_edges:
        final_graph.add_edge(*edge)
    assert is_isomorphic(br.generate_rearranged_graph(init_graph, [(3, 4)], [(1, 2)]), final_graph)


def test_2b():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 20, 0, 0)])

    # Reactants to products must break two bonds
    assert len(br.get_bond_rearrangs(ReactantComplex(reac), ProductComplex(prod), name='2b_test')) == 1
    os.remove('2b_test_bond_rearrangs.txt')

    assert br.get_fbonds_bbonds_2b(reac, prod, [], [[(0, 1), (1, 2)]], [], [], [(0, 2)], []) == [br.BondRearrangement(breaking_bonds=[(0, 1), (1, 2)])]


def test_3b():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0), Atom('H', 1.8, 0, 0)])
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 20, 0, 0), Atom('H', 30, 0, 0)])

    # Reactants to products must break three bonds but this is not yet supported in any form
    assert br.get_bond_rearrangs(ReactantComplex(reac), ProductComplex(prod), name='3b_test') is None


def test_1b1f():
    reac = Molecule(atoms=[Atom('C', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('C', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 10.6, 0, 0)])
    assert br.get_fbonds_bbonds_1b1f(reac, prod, [], [[(0, 1)]], [[(1, 2)]], [], [], []) == [br.BondRearrangement(forming_bonds=[(1, 2)], breaking_bonds=[(0, 1)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 10.6, 0, 0)])
    assert br.get_fbonds_bbonds_1b1f(reac, prod, [], [], [], [[[(0, 1)], [(1, 2)]]], [], []) == [br.BondRearrangement(forming_bonds=[(1, 2)], breaking_bonds=[(0, 1)])]


def test_2b1f():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('O', 1.2, 0, 0)])
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('O', 0.6, 0, 0)])
    assert br.get_fbonds_bbonds_2b1f(reac, prod, [], [[(0, 1)], [(1, 2)]], [[(0, 2)]], [], [], []) == [br.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 0.6, 0, 0)])
    assert br.get_fbonds_bbonds_2b1f(reac, prod, [], [[(0, 1), (1, 2)]], [[(0, 2)]], [], [], []) == [br.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    assert br.get_fbonds_bbonds_2b1f(reac, prod, [], [[(0, 1), (1, 2)]], [], [], [(0, 2)], []) == [br.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)])]


def test_2b2f():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('N', 10, 0, 0), Atom('O', 10.6, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('N', 0.6, 0, 0), Atom('O', 10.6, 0, 0)])

    assert br.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1)], [(2, 3)]], [[(0, 2)], [(1, 3)]], [], [], []) == [
        br.BondRearrangement(forming_bonds=[(0, 2), (1, 3)], breaking_bonds=[(0, 1), (2, 3)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 10, 0, 0),
                           Atom('N', 10.6, 0, 0), Atom('O', 20, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 1.2, 0, 0),
                           Atom('N', 20, 0, 0), Atom('O', 0.6, 0, 0)])

    assert br.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1)], [(2, 3)]], [[(0, 4), (2, 4)]], [], [], []) == [
        br.BondRearrangement(forming_bonds=[(0, 4), (2, 4)], breaking_bonds=[(0, 1), (2, 3)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 1.2, 0, 0), Atom('O', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 11.2, 0, 0), Atom('O', 10.6, 0, 0)])
    assert br.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1), (1, 2)]], [[(0, 3), (2, 3)], [(1, 3)]], [], [], []) == [
        br.BondRearrangement(forming_bonds=[(0, 3), (1, 3)], breaking_bonds=[(0, 1), (1, 2)]),
        br.BondRearrangement(forming_bonds=[(1, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 1.2, 0, 0), Atom('O', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 1.2, 0, 0), Atom('O', 0.6, 0, 0)])
    assert br.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1), (1, 2)]], [[(0, 3), (2, 3)]], [], [], []) == [
        br.BondRearrangement(forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('N', 1.2, 0, 0), Atom('C', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('N', 1.2, 0, 0), Atom('C', 0.6, 0, 0)])
    assert br.get_fbonds_bbonds_2b2f(reac, prod, [], [], [], [[[(0, 1)], [(0, 3)]], [[(1, 2)], [(2, 3)]]], [], []) == [br.BondRearrangement(forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)])]
