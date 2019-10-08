from autode import mol_graphs

xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0], ['H', 1.4, 0.0, 0.0]]
graph = mol_graphs.make_graph(xyz_list, 3)
xyzlist2 = [['H', 0.0, 0.0, 0.0], ['H', 0.3, 0.0, 0.0]]
graph2 = mol_graphs.make_graph(xyzlist2, 2)


def test_make_graph():
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    neighbours = [node for node in graph.neighbors(1)]
    assert neighbours == [0,2]


def test_isomorphism_subgraph():
    should_be_true = mol_graphs.is_subgraph_isomorphic(graph, graph2)
    assert should_be_true == True

    xyzlist3 = [['H', 0.0, 0.0, 0.0], ['H', 0.3, 0.0, 0.0], ['H', 0.6, 0.0, 0.0]]
    graph3 = mol_graphs.make_graph(xyzlist3, 2)
    should_be_false = mol_graphs.is_subgraph_isomorphic(graph, graph3)
    assert should_be_false == False


def test_graph_isomorphism():
    same_as_g1 = mol_graphs.make_graph(xyz_list, 3)
    should_be_true = mol_graphs.is_isomorphic(graph, same_as_g1)
    assert should_be_true == True

    should_be_false = mol_graphs.is_isomorphic(graph, graph2)
    assert should_be_false == False


def test_adjacency_digraph():
    digraph = mol_graphs.get_adjacency_digraph(1, graph)
    assert digraph.number_of_nodes() == 3
    assert digraph.number_of_edges() == 2
    assert digraph.is_directed() == True