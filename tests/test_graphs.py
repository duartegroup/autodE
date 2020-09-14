from autode import mol_graphs
from autode.bond_rearrangement import BondRearrangement
from autode.exceptions import NoMolecularGraph, NoMapping
from autode.species.species import Species
from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.conformers import Conformer
from autode.input_output import xyz_file_to_atoms
from . import testutils
import networkx as nx
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))

h_a = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
h_b = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.7)

h2 = Species(name='H2', atoms=[h_a, h_b], charge=0, mult=1)
mol_graphs.make_graph(h2)

g = nx.Graph()
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]
for edge in edges:
    g.add_edge(*edge)


def test_graph_generation():

    assert h2.graph.number_of_edges() == 1
    assert h2.graph.number_of_nodes() == 2
    assert h2.graph.nodes[0]['atom_label'] == 'H'


def test_edge_cases():
    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.6)

    # For H3 with a slightly longer bond on one side there should only be
    # 1 'bond'
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

    # H3 in a triangular arrangement should not be sub-graph isomorphic to linear H4
    h_e = Atom(atomic_symbol='H', x=0.3, y=0.0, z=0.3)
    h3 = Species(name='H_H', charge=0, mult=1, atoms=[h_a, h_b, h_e])
    mol_graphs.make_graph(h3, allow_invalid_valancies=True)
    assert mol_graphs.is_subgraph_isomorphic(larger_graph=h4.graph, smaller_graph=h3.graph) is False


def test_ts_template():

    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.4)

    ts_template = Species(name='template', charge=0, mult=1,
                          atoms=[h_a, h_b, h_c])
    mol_graphs.make_graph(species=ts_template, allow_invalid_valancies=True)
    ts_template.graph.edges[0, 1]['active'] = True

    ts = Species(name='template', charge=0, mult=1, atoms=[h_a, h_b, h_c])
    mol_graphs.make_graph(species=ts, allow_invalid_valancies=True)
    ts.graph.edges[1, 2]['active'] = True

    mapping = mol_graphs.get_mapping_ts_template(ts.graph, ts_template.graph)
    assert mapping is not None
    assert type(mapping) == dict

    assert mol_graphs.is_isomorphic(ts.graph, ts_template.graph, ignore_active_bonds=True)


def test_truncated_active_graph():

    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.4)
    h_d = Atom(atomic_symbol='H', x=0.0, y=0.0, z=2.1)

    ts = Species(name='template', charge=0, mult=1, atoms=[h_a, h_b, h_c, h_d])
    mol_graphs.make_graph(species=ts, allow_invalid_valancies=True)

    # H--active--H--H--H should truncate by keeping only the nearest neighbours to the first two atoms
    truncated_graph = mol_graphs.get_truncated_active_mol_graph(ts.graph, active_bonds=[(0, 1)])
    assert truncated_graph.number_of_nodes() == 3
    assert truncated_graph.number_of_edges() == 2


def test_mapping():
    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.4)

    h3_a = Species(name='template', charge=0, mult=1,
                          atoms=[h_a, h_b, h_c])
    mol_graphs.make_graph(species=h3_a, allow_invalid_valancies=True)

    h3_b = Species(name='template', charge=0, mult=1, atoms=[h_a, h_b, h_c])
    mol_graphs.make_graph(species=h3_b, allow_invalid_valancies=True)

    # Isomorphic (identical) graphs should have at least one mapping between them
    mapping = mol_graphs.get_mapping(h3_b.graph, h3_a.graph)
    assert mapping is not None
    assert type(mapping) == dict


def test_not_isomorphic():

    h_c = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.0)
    h2_b = Species(name='template', charge=0, mult=1, atoms=[h_a, h_c])
    mol_graphs.make_graph(species=h2_b, rel_tolerance=0.3)

    assert mol_graphs.is_isomorphic(h2.graph, h2_b.graph) is False


def test_not_isomorphic2():

    c = Atom(atomic_symbol='C', x=0.0, y=0.0, z=0.7)
    ch = Species(name='ch', atoms=[h_a, c], charge=0, mult=2)
    mol_graphs.make_graph(ch)

    assert mol_graphs.is_isomorphic(h2.graph, ch.graph) is False


def test_find_cycles():

    assert mol_graphs.find_cycles(g) == [[1, 2, 0]]


def test_reac_to_prods():

    rearrang = BondRearrangement([(0, 4)], [(3, 4)])
    prod_graph = mol_graphs.reac_graph_to_prod_graph(g, rearrang)
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


def test_species_isomorphism():

    h2.graph = None
    h2_copy = Species(name='H2', atoms=[h_a, h_b], charge=0, mult=1)
    h2_copy.graph = None

    with pytest.raises(NoMolecularGraph):
        assert mol_graphs.species_are_isomorphic(h2, h2_copy)

    # With molecular graphs the species should be isomorphic
    mol_graphs.make_graph(h2)
    mol_graphs.make_graph(h2_copy)

    assert mol_graphs.species_are_isomorphic(h2, h2_copy)

    # Shift one of the atoms far away and remake the graph
    h2_copy.atoms[1].translate(vec=np.array([10, 0, 0]))
    mol_graphs.make_graph(h2_copy)

    assert mol_graphs.species_are_isomorphic(h2, h2_copy) is False

    # Generating a pair of conformers that are isomporhpic should return that
    # the species are again isomorphic
    h2.conformers = [Conformer(name='h2_conf', atoms=[h_a, h_b], charge=0, mult=1)]
    h2_copy.conformers = [Conformer(name='h2_conf', atoms=[h_a, h_b], charge=0, mult=1)]

    assert mol_graphs.species_are_isomorphic(h2, h2_copy)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'e2_tss.zip'))
def test_isomorphic_no_active():

    ts_syn = Conformer(name='syn_ts', charge=-1, mult=0,
                       atoms=xyz_file_to_atoms('E2_ts_syn.xyz'))
    mol_graphs.make_graph(ts_syn)
    mol_graphs.set_active_mol_graph(ts_syn, active_bonds=[(8, 5), (0, 5), (1, 2)])

    ts_anti = Conformer(name='anti_ts', charge=-1, mult=0,
                        atoms=xyz_file_to_atoms('E2_ts.xyz'))
    mol_graphs.make_graph(ts_anti)

    assert mol_graphs.is_isomorphic(ts_syn.graph, ts_anti.graph,
                                    ignore_active_bonds=True)


def test_timeout():

    # Generate a large-ish graph
    graph = nx.Graph()
    for i in range(10000):
        graph.add_node(i)

    for _ in range(5000):
        (i, j) = np.random.randint(0, 1000, size=2)

        if (i, j) not in graph.edges:
            graph.add_edge(i, j)

    node_perm = np.random.permutation(list(graph.nodes))
    mapping = {u: v for (u, v) in zip(graph.nodes, node_perm)}

    isomorphic_graph = nx.relabel_nodes(graph, mapping=mapping, copy=True)

    # With a short timeout this should return False - not sure this is the
    # optimal behavior
    assert not mol_graphs.is_isomorphic(graph, isomorphic_graph, timeout=1)


def test_species_conformers_isomorphic():
    h2_a = Molecule(name='H2', atoms=[Atom('H'), Atom('H', x=0.7)])

    h2_b = Molecule(name='H2', atoms=[Atom('H'), Atom('H', x=1.5)])

    assert not mol_graphs.species_are_isomorphic(h2_a, h2_b)

    # Should raise an exception for two non-isomorphic graphs
    with pytest.raises(NoMapping):
        mol_graphs.get_mapping(h2_a.graph, h2_b.graph)

    h2_a.conformers = None
    h2_b.conformers = [Conformer(name='H2', atoms=[Atom('H'), Atom('H', x=0.7)])]

    assert mol_graphs.species_are_isomorphic(h2_a, h2_b)


def test_graph_without_active_edges():

    mol = Molecule(name='H2', atoms=[Atom('H'), Atom('H', x=0.7)])
    mol.graph.edges[(0, 1)]['active'] = True

    graph = mol_graphs.get_graph_no_active_edges(mol.graph)
    # Should now have no edges if the one bond was defined as active
    assert graph.number_of_edges() == 0

