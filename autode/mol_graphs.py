from autode.log import logger
import networkx as nx
import multiprocessing as mp
from networkx.algorithms import isomorphism
from scipy.spatial import distance_matrix
from autode.bond_lengths import get_avg_bond_length
from autode.atoms import is_pi_atom


def make_graph(species, rel_tolerance=0.2, rdkit_bonds=None):
    """
    Make the molecular graph from the 'bonds' determined on a distance criteria or a list of RDKit bonds .
    No distinction is made between single, double etc. bond types.

    Arguments:
        species (autode.species.Species):

    Keyword Arguments:
        rel_tolerance (float):
        rdkit_bonds (RDKit.Chem.Bond):
    """
    logger.info('Generating molecular graph with NetworkX')

    graph = nx.Graph()

    # Add the atoms to the graph all are initially assumed not to be stereocenters
    for i in range(species.n_atoms):
        graph.add_node(i, atom_label=species.atoms[i].label, stereo=False)

    # If rdkit bonds object is specified then add edges to the graph and return
    if rdkit_bonds is not None:
        [graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), pi=False) for bond in rdkit_bonds]
        species.graph = graph
        return None

    # Loop over the unique pairs of atoms and add 'bonds'
    coordinates = species.get_coordinates()
    dist_mat = distance_matrix(coordinates, coordinates)

    for i in range(species.n_atoms):
        for j in range(i + 1, species.n_atoms):

            avg_bond_length = get_avg_bond_length(atom_i_label=species.atoms[i].label,
                                                  atom_j_label=species.atoms[j].label)

            # If the distance between atoms i and j are less or equal to 1.2x average length add a 'bond'
            if dist_mat[i, j] <= avg_bond_length * (1.0 + rel_tolerance):
                graph.add_edge(i, j, pi=False)

    species.graph = graph
    set_pi_bonds(species)

    return None


def set_pi_bonds(species):
    """
    For a molecular species set the π bonds in the molecular graph.

    Arguments:
        species (autode.species.Species):
    """
    logger.info('Setting the π bonds in a species')

    for bond in species.graph.edges:
        atom_i, atom_j = bond

        if all([is_pi_atom(atom_label=species.atoms[atom].label, valency=species.graph.degree[atom]) for atom in bond]):
            species.graph.edges[atom_i, atom_j]['pi'] = True

    return None


def union(graphs):
    """Return the union of two graphs. The disjoint union is returned"""
    return nx.disjoint_union_all(graphs)


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
    logger.info('Running isomorphism')
    graph_matcher = isomorphism.GraphMatcher(larger_graph, smaller_graph,
                                             node_match=isomorphism.categorical_node_match('atom_label', 'C'),
                                             edge_match=isomorphism.categorical_edge_match('active', False))
    return next(graph_matcher.match())


def get_mapping(graph, other_graph):
    """Return a sorted mapping"""

    logger.info('Running isomorphism')
    gm = isomorphism.GraphMatcher(graph, other_graph,
                                  node_match=isomorphism.categorical_node_match('atom_label', 'C'))

    mapping = next(gm.match())
    return {i: mapping[i] for i in sorted(mapping)}


def reorder_nodes(graph, mapping):
    # NetworkX uses the inverse mapping so the dict is swapped before the nodes are relabeled
    return nx.relabel_nodes(graph, mapping={u: v for v, u in mapping.items()}, copy=True)


def get_active_mol_graph(species, active_bonds):
    logger.info('Getting molecular graph with active edges')
    active_graph = species.graph.copy()

    for bond in active_bonds:
        has_edge = False
        for edge in active_graph.edges():
            if edge == bond or edge == tuple(reversed(bond)):
                has_edge = True
                active_graph.edges[edge[0], edge[1]]['active'] = True

        if not has_edge:
            active_graph.add_edge(*bond, active=True)

    return active_graph


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


def find_cycles(graph):
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
        reac_graph (nx.Graph): graph of the reactant
        bond_rearrang (bond rearrang object): the bond rearrang linking reacs and prods

    Returns:
        nx.Graph: graph of the product with each atom indexed as in the reactants
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


def split_mol_across_bond(graph, bond):
    """Gets a list of atoms on either side of a bond

    Arguments:
        graph (nx.graph): molecular graph
        bond (tuple): list of bonds to be split across

    """
    graph_copy = graph.copy()

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
