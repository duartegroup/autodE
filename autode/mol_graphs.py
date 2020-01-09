from autode.log import logger
import networkx as nx
import multiprocessing as mp
from networkx.algorithms import isomorphism
from autode.bond_lengths import get_xyz_bond_list


def make_graph(xyzs, n_atoms, bonds=None):
    logger.info('Generating molecular graph with networkx')

    graph = nx.Graph()
    for i in range(n_atoms):
        graph.add_node(i, atom_label=xyzs[i][0])

    if bonds is None:
        bonded_atom_list = get_xyz_bond_list(xyzs)
    else:
        bonded_atom_list = bonds
        
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
    # hopefully the first one is fine(?)
    return get_mapping(larger_graph, smaller_graph, graph_matcher)[0]


def get_mapping(larger_graph, smaller_graph, graph_matcher=None):
    logger.info('Running subgraph isomorphism')
    if graph_matcher is None:
        graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                                 node_match=isomorphism.categorical_node_match('atom_label', 'C'))

    return [subgraph for subgraph in graph_matcher.subgraph_isomorphisms_iter()]


def is_isomorphic(graph1, graph2):
    """Check whether two NX graphs are isomorphic. Contains a timeout because the gm.is_isomorphic() method is found
    to ocassionaly get stuck

    Arguments:
        graph1 (nx.Graph): graph 1
        graph2 (nx.Graph): graph 2

    Returns:
        bool: if the graphs are isomorphic
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
        graph (nx.Graph): the molecular graph

    Returns:
        list(list): each list has the atoms in a cycle
    """
    return nx.cycle_basis(graph)


def reac_graph_to_prods(reac_graph, bond_rearrang):
    """Makes the graph of the product from the reactant and the bond rearrang, so it has the indices of the reactant

    Arguments:
        reac_graph (nx.graph): graph of the reactant
        bond_rearrang (bond rearrang object): the bond rearrang linking reacs and prods

    Returns:
        nx.graph: graph of the product with each atom indexed as in the reactants
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
        graph (nx.graph): graph

    Returns:
        list: list of graphs separate graphs
    """
    return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]


def split_mol_across_bond(graph, bonds):
    """gets a list of atoms on either side of a bond

    Arguments:
        graph (nx.graph): molecular graph
        bond (ist): list of bonds to be split across

    """
    graph_copy = graph.copy()
    for bond in bonds:
        graph_copy.remove_edge(*bond)
    split_subgraphs = get_separate_subgraphs(graph_copy)
    return [list(graph.nodes) for graph in split_subgraphs]


def get_bond_type_list(graph):
    """Finds the types (i.e CH) of all the bonds in a molecular graph

    Arguments:
        graph (nx.Graph): Molecular graph

    Returns:
        bond_list_dict (dict): key = bond type, value = list of bonds of this type
    """
    bond_list_dict = {}
    atom_types = set()

    for _, atom_label in graph.nodes.data('atom_label'):
        atom_types.add(atom_label)

    ordered_atom_labels = sorted(atom_types)

    for index, atom_label in enumerate(ordered_atom_labels):
        for i in range(index, len(ordered_atom_labels)):
            key = atom_label + ordered_atom_labels[i]
            bond_list_dict[key] = []

    for bond in graph.edges:
        atom_i_label = graph.nodes[bond[0]]['atom_label']
        atom_j_label = graph.nodes[bond[1]]['atom_label']
        key1, key2 = atom_i_label + atom_j_label, atom_j_label + atom_i_label

        if key1 in bond_list_dict.keys():
            bond_list_dict[key1].append(bond)
        elif key2 in bond_list_dict.keys():
            bond_list_dict[key2].append(bond)

    return bond_list_dict


def get_fbonds(graph, key):
    """Get all the possible forming bonds of a certain type
    
    Arguments:
        graph (nx.Graph): graph object of a molecule
        key (str): string representing the bond type to be examined
    
    Returns:
        list: list of bonds that can be made of this type
    """
    possible_fbonds = []
    bonds = list(graph.edges)
    for atom_i in graph.nodes:
        for atom_j in graph.nodes:
            if atom_i < atom_j:
                if not (atom_i, atom_j) in bonds and not (atom_j, atom_i) in bonds:
                    bond = (atom_i, atom_j)
                    atom_i_label = graph.nodes[bond[0]]['atom_label']
                    atom_j_label = graph.nodes[bond[1]]['atom_label']
                    key1, key2 = atom_i_label + atom_j_label, atom_j_label + atom_i_label
                    if key1 == key or key2 == key:
                        possible_fbonds.append(bond)

    return possible_fbonds
