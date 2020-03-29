from autode import bond_rearrangement as rearr
from autode.molecule import Molecule
from autode.complex import ReactantComplex, ProductComplex
from autode.atoms import Atom
from autode.mol_graphs import is_isomorphic
import networkx as nx
import os


def test_bondrearr_obj():
    # Reaction H + H2 -> H2 + H
    rearrang = rearr.BondRearrangement(forming_bonds=[(0, 1)],
                                       breaking_bonds=[(1, 2)])

    assert rearrang.n_fbonds == 1
    assert rearrang.n_bbonds == 1

    rearrag2 = rearr.BondRearrangement(forming_bonds=[(0, 1)],
                                       breaking_bonds=[(1, 2)])
    assert rearrag2 == rearrang

    h2_h_mol = Molecule(name='mol', atoms=[Atom('H', 0.0, 0.0, 0.0),
                                           Atom('H', 0.0, 0.0, -1.0),
                                           Atom('H', 0.0, 0.0, 1.0)])

    active_atom_nl = rearrang.get_active_atom_neighbour_lists(mol=h2_h_mol, depth=1)
    assert len(active_atom_nl) == 4
    assert active_atom_nl == [['H'], ['H'], ['H'], ['H']]


def test_get_bond_rearrangs():
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

    assert rearr.get_bond_rearrangs(ReactantComplex(reac), ProductComplex(prod), name='test') == [rearr.BondRearrangement(breaking_bonds=[(0, 1)])]
    assert rearr.get_bond_rearrangs(ReactantComplex(prod), ProductComplex(reac), name='test2') is None

    os.remove('test_bond_rearrangs.txt')


def test_add_bond_rearrang():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0)])
    assert rearr.add_bond_rearrangment([], reac, prod, [], [(0, 1)]) == [rearr.BondRearrangement(breaking_bonds=[(0, 1)])]


def test_generate_rearranged_graph():
    init_graph = nx.Graph()
    final_graph = nx.Graph()
    init_edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    final_edges = [(0, 1), (2, 3), (3, 4), (4, 5), (5, 6)]
    for edge in init_edges:
        init_graph.add_edge(*edge)
    for edge in final_edges:
        final_graph.add_edge(*edge)
    assert is_isomorphic(rearr.generate_rearranged_graph(init_graph, [(3, 4)], [(1, 2)]), final_graph)


def test_2b():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 20, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b(reac, prod, [], [[(0, 1), (1, 2)]], [], [], [(0, 2)], []) == [rearr.BondRearrangement(breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('C', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    prod = Molecule(atoms=[Atom('C', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 20, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b(reac, prod, [], [[(0, 1)], [(1, 2)]], [], [], [], []) == [rearr.BondRearrangement(breaking_bonds=[(0, 1), (1, 2)])]


def test_1b1f():
    reac = Molecule(atoms=[Atom('C', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('C', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 10.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_1b1f(reac, prod, [], [[(0, 1)]], [[(1, 2)]], [], [], []) == [rearr.BondRearrangement(forming_bonds=[(1, 2)], breaking_bonds=[(0, 1)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 10, 0, 0), Atom('H', 10.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_1b1f(reac, prod, [], [], [], [[[(0, 1)], [(1, 2)]]], [], []) == [rearr.BondRearrangement(forming_bonds=[(1, 2)], breaking_bonds=[(0, 1)])]


def test_1b2f():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('O', 0.6, 0, 0), Atom('C', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('O', 1.2, 0, 0), Atom('C', 0.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_1b2f(reac, prod, [], [[(0, 1)]], [[(0, 2)], [(1, 2)]], [], [], []) == [rearr.BondRearrangement(forming_bonds=[(0, 2), (1, 2)], breaking_bonds=[(0, 1)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('C', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 1.2, 0, 0), Atom('C', 0.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_1b2f(reac, prod, [], [[(0, 1)]], [[(0, 2), (1, 2)]], [], [], []) == [rearr.BondRearrangement(forming_bonds=[(0, 2), (1, 2)], breaking_bonds=[(0, 1)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    assert rearr.get_fbonds_bbonds_1b2f(reac, prod, [], [], [[(0, 2), (1, 2)]], [], [], [(0, 1)]) == [rearr.BondRearrangement(forming_bonds=[(0, 2), (1, 2)], breaking_bonds=[(0, 1)])]


def test_2b1f():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('O', 1.2, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('O', 0.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b1f(reac, prod, [], [[(0, 1)], [(1, 2)]], [[(0, 2)]], [], [], []) == [rearr.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 0.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b1f(reac, prod, [], [[(0, 1), (1, 2)]], [[(0, 2)]], [], [], []) == [rearr.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 1.2, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('H', 0.6, 0, 0), Atom('H', 10, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b1f(reac, prod, [], [[(0, 1), (1, 2)]], [], [], [(0, 2)], []) == [rearr.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)])]


def test_2b2f():
    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('N', 10, 0, 0), Atom('O', 10.6, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('N', 0.6, 0, 0), Atom('O', 10.6, 0, 0)])

    assert rearr.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1)], [(2, 3)]], [[(0, 2)], [(1, 3)]], [], [], []) == [
        rearr.BondRearrangement(forming_bonds=[(0, 2), (1, 3)], breaking_bonds=[(0, 1), (2, 3)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 10, 0, 0),
                           Atom('N', 10.6, 0, 0), Atom('O', 20, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 1.2, 0, 0),
                           Atom('N', 20, 0, 0), Atom('O', 0.6, 0, 0)])

    assert rearr.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1)], [(2, 3)]], [[(0, 4), (2, 4)]], [], [], []) == [
        rearr.BondRearrangement(forming_bonds=[(0, 4), (2, 4)], breaking_bonds=[(0, 1), (2, 3)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 1.2, 0, 0), Atom('O', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 11.2, 0, 0), Atom('O', 10.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1), (1, 2)]], [[(0, 3), (2, 3)], [(1, 3)]], [], [], []) == [
        rearr.BondRearrangement(forming_bonds=[(0, 3), (1, 3)], breaking_bonds=[(0, 1), (1, 2)]),
        rearr.BondRearrangement(forming_bonds=[(1, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('H', 1.2, 0, 0), Atom('O', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('H', 1.2, 0, 0), Atom('O', 0.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b2f(reac, prod, [], [[(0, 1), (1, 2)]], [[(0, 3), (2, 3)]], [], [], []) == [
        rearr.BondRearrangement(forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)])]

    reac = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 0.6, 0, 0), Atom('N', 1.2, 0, 0), Atom('C', 10, 0, 0)])
    prod = Molecule(atoms=[Atom('H', 0, 0, 0), Atom('C', 10, 0, 0), Atom('N', 1.2, 0, 0), Atom('C', 0.6, 0, 0)])
    assert rearr.get_fbonds_bbonds_2b2f(reac, prod, [], [], [], [[[(0, 1)], [(0, 3)]], [[(1, 2)], [(2, 3)]]], [], []) == [rearr.BondRearrangement(forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)])]
