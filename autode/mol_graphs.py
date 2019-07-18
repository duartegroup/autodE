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


def is_subgraph_isomorphic(larger_graph, smaller_graph):
    logger.info('Running subgraph isomorphism')
    graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                             node_match=isomorphism.categorical_node_match('atom_label', 'C'),
                                             edge_match=isomorphism.categorical_edge_match('active', False))
    if graph_matcher.subgraph_is_isomorphic():
        return True

    return False


def get_mapping_ts_template(larger_graph, smaller_graph):
    logger.info('Getting mapping of molecule onto the TS template')
    graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                             node_match=isomorphism.categorical_node_match('atom_label', 'C'),
                                             edge_match=isomorphism.categorical_edge_match('active', False))
    return get_mapping(larger_graph, smaller_graph, graph_matcher)[0]       # hopefully the first one is fine(?)


def get_mapping(larger_graph, smaller_graph, graph_matcher=None):
    logger.info('Running subgraph isomorphism')
    if graph_matcher is None:
        graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                                 node_match=isomorphism.categorical_node_match('atom_label', 'C'))

    return [subgraph for subgraph in graph_matcher.subgraph_isomorphisms_iter()]


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


def get_adjacency_graph(atom_i, graph):

    added_atoms = []
    adj_graph = nx.Graph()

    def add_adj_nodes(atom_k):
        adj_graph.add_node(atom_k, atom_label=graph.nodes[atom_k]['atom_label'])

        print(adj_graph.nodes.data())

        if adj_graph.number_of_nodes() == graph.number_of_nodes():
            print('returning')
            return adj_graph

        adj_atoms = list(graph.adj[atom_k])
        print(atom_k, adj_atoms)
        for atom_j in adj_atoms:
            print(atom_k, atom_j)
            if atom_j not in added_atoms:
                adj_graph.add_node(atom_j, atom_label=graph.nodes[atom_j]['atom_label'])
                adj_graph.add_edge(atom_k, atom_j)
                added_atoms.append(atom_j)

                return add_adj_nodes(atom_j)
            else:
                pass

    adj_graph = add_adj_nodes(atom_i)
    exit()
    return adj_graph
