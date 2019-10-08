from autode import mol_graphs

h2_xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]]


def test_graph_generation():

    h2_graph = mol_graphs.make_graph(h2_xyzs, n_atoms=len(h2_xyzs))
    assert h2_graph.number_of_edges() == 1
    assert h2_graph.number_of_nodes() == 2
    assert h2_graph.nodes[0]['atom_label'] == 'H'


def test_isomorphic_graphs():

    h2_graph1 = mol_graphs.make_graph(h2_xyzs, n_atoms=len(h2_xyzs))
    h2_graph2 = mol_graphs.make_graph(list(reversed(h2_xyzs)), n_atoms=len(h2_xyzs))

    assert mol_graphs.is_isomorphic(h2_graph1, h2_graph2) is True


def test_subgraph_isomorphism():

    h2_graph = mol_graphs.make_graph(h2_xyzs, n_atoms=len(h2_xyzs))
    h4_graph = mol_graphs.make_graph(h2_xyzs*2, n_atoms=len(h2_xyzs)*2)

    assert mol_graphs.is_subgraph_isomorphic(larger_graph=h4_graph, smaller_graph=h2_graph) is True

    h2_h4_mapping = mol_graphs.get_mapping_ts_template(larger_graph=h4_graph, smaller_graph=h2_graph)
    assert type(h2_h4_mapping) == dict
    assert len(h2_h4_mapping) == 2
