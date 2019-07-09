from .log import logger
import networkx as nx
from networkx.algorithms import isomorphism
from .bond_lengths import get_xyz_bond_list


def make_graph(xyzs, n_atoms):
    logger.info('Generating molecular graph with networkx')

    graph = nx.Graph()
    for i in range(n_atoms):
        graph.add_node(i, atom_label=xyzs[i][0])

    bonded_atom_list = get_xyz_bond_list(xyzs)
    for pair in bonded_atom_list:
        graph.add_edge(*pair)

    return graph


def get_mapping(larger_graph, smaller_graph):
    logger.info('Running subgraph isomorphism')
    graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                             node_match=isomorphism.categorical_node_match('atom_label', 'C'))
    isomorphic_subgraphs = []

    for subgraph in graph_matcher.subgraph_isomorphisms_iter():
        isomorphic_subgraphs.append(subgraph)

    return isomorphic_subgraphs


def is_isomorphic(graph1, graph2):
    if isomorphism.faster_could_be_isomorphic(graph1, graph2):
        graph_matcher = isomorphism.GraphMatcher(graph1, graph2,
                                                 node_match=isomorphism.categorical_node_match('atom_label', 'C'))
        return graph_matcher.is_isomorphic()
    else:
        return False


def get_isomorphic_mapping(graph1, graph2):

    graph_matcher = isomorphism.GraphMatcher(graph1, graph2,
                                             node_match=isomorphism.categorical_node_match('atom_label', 'C'))
    if graph_matcher.is_isomorphic():
        return graph_matcher.mapping
    else:
        logger.warning('Graphs were not isomorphic')
        return None
