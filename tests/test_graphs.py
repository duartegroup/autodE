from autode import mol_graphs


def test_graph_generation():
    xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]]

    h2_graph = mol_graphs.make_graph(xyzs, n_atoms=len(xyzs))
    assert h2_graph.number_of_edges() == 1
    assert h2_graph.number_of_nodes() == 2
    assert h2_graph.nodes[0]['atom_label'] == 'H'
