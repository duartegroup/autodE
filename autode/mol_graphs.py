from .log import logger
import networkx as nx
import multiprocessing as mp
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
    """
    Check whether two NX graphs are isomorphic. Contains a timeout because the gm.is_isomorphic() method is found
    to ocassionaly get stuck

    :param graph1: (object) nx graph
    :param graph2: (object) nx graph
    :return:
    """

    if isomorphism.faster_could_be_isomorphic(graph1, graph2):
        graph_matcher = isomorphism.GraphMatcher(graph1, graph2,
                                                 node_match=isomorphism.categorical_node_match('atom_label', 'C'))

        manager = mp.Manager()
        res = manager.dict()
        p = mp.Process(target=gm_is_isomorphic, args=(graph_matcher, res))
        p.start()
        p.join(5)

        if p.is_alive():
            p.terminate()
            logger.error('NX graph matching hanging')
            return False

        return res.values()[0]
    else:
        return False


def gm_is_isomorphic(gm, result):
    result[0] = gm.is_isomorphic()


def get_isomorphic_mapping(graph1, graph2):

    graph_matcher = isomorphism.GraphMatcher(graph1, graph2,
                                             node_match=isomorphism.categorical_node_match('atom_label', 'C'))
    if graph_matcher.is_isomorphic():
        return graph_matcher.mapping
    else:
        logger.warning('Graphs were not isomorphic')
        return None


def get_adjacency_digraph(atom_i, graph):
    """
    Make a directed (di) graph centred on atom_i with it's neighbours and neighbours neighbours etc. then break when
    we've added all the nodes, but don't add any loops etc. Used to determine if
    :param atom_i:
    :param graph:
    :return:
    """

    adj_graph = nx.DiGraph()
    added_atoms = []

    def add_adj_nodes(atom_k):
        if atom_k not in added_atoms:
            added_atoms.append(atom_k)
        adj_graph.add_node(atom_k, atom_label=graph.nodes[atom_k]['atom_label'])

        if adj_graph.number_of_nodes() == graph.number_of_nodes():
            return None

        adj_atoms = list(graph.adj[atom_k])
        unique_adj_atoms = []
        for atom_j in adj_atoms:
            if atom_j not in added_atoms:
                adj_graph.add_node(atom_j, atom_label=graph.nodes[atom_j]['atom_label'])
                adj_graph.add_edge(atom_k, atom_j)
                added_atoms.append(atom_j)
                unique_adj_atoms.append(atom_j)
                [add_adj_nodes(atom) for atom in unique_adj_atoms]

    add_adj_nodes(atom_i)

    return adj_graph
