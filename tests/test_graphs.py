from autode import mol_graphs
from autode.bond_rearrangement import BondRearrangement
from autode.species import Species
from autode.atoms import Atom
import networkx as nx

h_a = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
h_b = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.7)

h2 = Species(name='H2', atoms=[h_a, h_b], charge=0, mult=1)
mol_graphs.make_graph(h2)


def test_graph_generation():

    assert h2.graph.number_of_edges() == 1
    assert h2.graph.number_of_nodes() == 2
    assert h2.graph.nodes[0]['atom_label'] == 'H'


def test_edge_cases():
    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.6)

    # For H3 with a slightly longer bond on one side there should only be 1 'bond'
    h3 = Species(name='H2', atoms=[h_a, h_b, h_c], charge=0, mult=1)
    mol_graphs.make_graph(h3)

    assert h3.graph.number_of_edges() == 1
    assert h3.graph.number_of_nodes() == 3


def test_remove_bonds():

    b3h6 = Species(name='diborane', charge=0, mult=1,
                   atoms=[Atom('B', -1.97106,  0.36170, -0.23984),
                          Atom('H', -0.91975, -0.06081,  0.43901),
                          Atom('H', -2.14001, -0.24547, -1.26544),
                          Atom('H', -2.99029,  0.31275,  0.39878),
                          Atom('B', -0.49819,  1.17500,  0.23984),
                          Atom('H',  0.52102,  1.22392, -0.39880),
                          Atom('H', -0.32919,  1.78217,  1.26543),
                          Atom('H', -1.54951,  1.59751, -0.43898)])

    mol_graphs.make_graph(species=b3h6)
    assert b3h6.graph.number_of_edges() == 6
    assert b3h6.graph.number_of_nodes() == 8

    # Boron atoms should be 3 fold valent
    assert len(list(b3h6.graph.neighbors(0))) == 3
    assert len(list(b3h6.graph.neighbors(4))) == 3


def test_isomorphic_graphs():
    h2_alt = Species(name='H2', atoms=[h_b, h_a], charge=0, mult=1)
    mol_graphs.make_graph(h2_alt)

    assert mol_graphs.is_isomorphic(h2.graph, h2_alt.graph) is True


def test_subgraph_isomorphism():

    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.4)
    h_d = Atom(atomic_symbol='H', x=0.0, y=0.0, z=2.1)
    h4 = Species(name='H4', atoms=[h_a, h_b, h_c, h_d], charge=0, mult=1)
    mol_graphs.make_graph(h4)

    assert mol_graphs.is_subgraph_isomorphic(larger_graph=h4.graph, smaller_graph=h2.graph) is True


def test_not_isomorphic():

    c = Atom(atomic_symbol='C', x=0.0, y=0.0, z=0.7)
    ch = Species(name='ch', atoms=[h_a, c], charge=0, mult=2)
    mol_graphs.make_graph(ch)

    assert mol_graphs.is_isomorphic(h2.graph, ch.graph) is False


g = nx.Graph()
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]
for edge in edges:
    g.add_edge(*edge)


def test_find_cycles():

    assert mol_graphs.find_cycles(g) == [[1, 2, 0]]


def test_reac_to_prods():

    rearrang = BondRearrangement([(0, 4)], [(3, 4)])
    prod_graph = mol_graphs.reac_graph_to_prods(g, rearrang)
    expected_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (0, 4)]
    expected_graph = nx.Graph()
    for edge in expected_edges:
        expected_graph.add_edge(*edge)

    assert mol_graphs.is_isomorphic(expected_graph, prod_graph)


def test_split_graph():
    assert mol_graphs.split_mol_across_bond(g, bond=(0, 3)) == [[0, 1, 2], [3, 4]]


def test_set_pi_bonds():

    ethene = Species(name='ethene', charge=0, mult=1,
                     atoms=[Atom('C', -2.20421, 0.40461, 0.00000),
                            Atom('C', -0.87115, 0.38845, 0.00000),
                            Atom('H', -2.76098, -0.22576, 0.68554),
                            Atom('H', -2.74554, 1.04829, -0.68554),
                            Atom('H', -0.32982, -0.25523, 0.68554),
                            Atom('H', -0.31437, 1.01882, -0.68554)])
    mol_graphs.make_graph(ethene)

    assert ethene.graph.edges[0, 1]['pi'] is True
    assert ethene.graph.edges[1, 0]['pi'] is True
    assert ethene.graph.edges[0, 2]['pi'] is False

    acetylene = Species(name='acetylene', charge=0, mult=1,
                        atoms=[Atom('C', -2.14031, 0.40384, 0.00000),
                               Atom('C', -0.93505, 0.38923, 0.00000),
                               Atom('H', -3.19861, 0.41666, 0.00000),
                               Atom('H',  0.12326, 0.37640, 0.00000)])
    mol_graphs.make_graph(acetylene)

    assert acetylene.graph.edges[0, 1]['pi'] is True
    assert acetylene.graph.edges[0, 2]['pi'] is False
