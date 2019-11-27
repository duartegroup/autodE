from autode.log import logger
import networkx as nx
import multiprocessing as mp
from networkx.algorithms import isomorphism
from autode.bond_lengths import get_xyz_bond_list


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
                                             node_match=isomorphism.categorical_node_match(
                                                 'atom_label', 'C'),
                                             edge_match=isomorphism.categorical_edge_match('active', False))
    if graph_matcher.subgraph_is_isomorphic():
        return True

    return False


def get_mapping_ts_template(larger_graph, smaller_graph):
    logger.info('Getting mapping of molecule onto the TS template')
    graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                             node_match=isomorphism.categorical_node_match(
                                                 'atom_label', 'C'),
                                             edge_match=isomorphism.categorical_edge_match('active', False))
    # hopefully the first one is fine(?)
    return get_mapping(larger_graph, smaller_graph, graph_matcher)[0]


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


def find_cycle(graph):
    """Finds all the cycles in a graph

    Arguments:
        graph {nx.Graph} -- the molecular graph

    Returns:
        list of list -- each list has the atoms in a cycle
    """
    return nx.cycle_basis(graph)


def reac_graph_to_prods(reac_graph, bond_rearrang):
    """Makes the graph of the product from the reactant and the bond rearrang, so it has the indices of the reactant

    Arguments:
        reac_graph {nx.graph} -- graph of the reactant
        bond_rearrang {bond rearrang object} -- the bond rearrang linking reacs and prods

    Returns:
        nx.graph -- graph of the product with each atom indexed as in the reactants
    """
    prod_graph = reac_graph.copy()
    for fbond in bond_rearrang.fbonds:
        prod_graph.add_edge(*fbond)
    for bbond in bond_rearrang.bbonds:
        prod_graph.remove_edge(*bbond)
    return prod_graph


def get_separate_subgraphs(graph):
    """Find all the unconnected graphs in a graph

    Arguments:
        graph {nx.graph} -- graph

    Returns:
        list -- list of graphs separate graphs
    """
    return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]


def split_mol_across_bond(graph, bonds, return_graphs=False):
    """gets a list of atoms on either side of a bond, or the two subgraphs if return_graphs=True

    Arguments:
        graph {nx.graph} -- molecular graph
        bond {list} -- list of bonds to be split across
        return_graphs {bool} -- true if want graphs instead of list of nodes
    """
    graph_copy = graph.copy()
    for bond in bonds:
        graph_copy.remove_edge(*bond)
    split_subgraphs = get_separate_subgraphs(graph_copy)
    if return_graphs:
        return split_subgraphs
    return [list(graph.nodes) for graph in split_subgraphs]


def get_pathway(graph, start_atom, end_atom):
    """Gets all the pathways from one atom to another

    Arguments:
        graph {nx.graph} -- mol graph
        start_atom {int} -- index of the starting atom node
        end_atom {int} -- index of the ending atom node

    Returns:
        list -- list of all simple pathways from start to end atoms
    """
    path_edges_list = []
    path_list = list(nx.all_simple_paths(graph, start_atom, end_atom))
    for path in map(nx.utils.pairwise, path_list):
        path_edges_list.append(list(path))
    return path_edges_list


def get_product_core_atoms(prod_mol, stripped_prod_graph):
    """Maps the

    Arguments:
        prod_mol {[type]} -- [description]
        stripped_prod_graph {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # have to remove H's from stripped graph as they won't be there in the whole mol
    for i in list(stripped_prod_graph.nodes):
        if stripped_prod_graph.nodes[i]['atom_label'] == 'H':
            stripped_prod_graph.remove_node(i)
    mapping_dict = get_mapping(prod_mol.graph, stripped_prod_graph)[0]
    return list(mapping_dict.keys())
