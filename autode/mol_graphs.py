from copy import deepcopy
import itertools
import multiprocessing as mp
from networkx.algorithms import isomorphism
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from autode.atoms import get_maximal_valance
from autode.atoms import is_pi_atom
from autode.bond_lengths import get_avg_bond_length
from autode.calculation import Calculation
from autode.log import logger
from autode.exceptions import CannotSplitAcrossBond
from autode.exceptions import NoMolecularGraph
from autode.atoms import get_atomic_weight
from autode.methods import get_lmethod
from autode.units import KcalMol


def make_graph(species, rel_tolerance=0.25, bond_list=None, allow_invalid_valancies=False):
    """
    Make the molecular graph from the 'bonds' determined on a distance criteria or a smiles parser object. All attributes
    default to false

    Nodes attributes:
        (0) atom_label: Atomic symbol of this atom
        (1) stereo: Is this atom part of some stereochemistry e.g. R/S or E/Z

    Edge attributes:
        (1) pi: Is this bond a pi bond. If it is then there should be no rotation the bond axis in conformer generation
        (2) active: Is this bond being made/broken (applies only to TransitionState objects)
    Arguments:

        species (autode.species.Species):

    Keyword Arguments:
        rel_tolerance (float):
        bond_list (list(tuple)):
        allow_invalid_valancies (bool):
    """
    logger.info('Generating molecular graph with NetworkX')

    graph = nx.Graph()

    # Add the atoms to the graph all are initially assumed not to be stereocenters
    for i in range(species.n_atoms):
        graph.add_node(i, atom_label=species.atoms[i].label, stereo=False)

    # If bonds are specified then add edges to the graph and return
    if bond_list is not None:
        [graph.add_edge(bond[0], bond[1], pi=False, active=False) for bond in bond_list]
        species.graph = graph
        return None

    else:
        # Loop over the unique pairs of atoms and add 'bonds'
        coordinates = species.get_coordinates()
        dist_mat = distance_matrix(coordinates, coordinates)

        for i in get_atom_ids_sorted_type(species):

            # Iterate through the closest atoms to atom i
            for j in np.argsort(dist_mat[i]):

                if i == j:
                    # Don't bond atoms to themselves
                    continue

                avg_bond_length = get_avg_bond_length(atom_i_label=species.atoms[i].label,
                                                      atom_j_label=species.atoms[j].label)

                # If the distance between atoms i and j are less or equal to 1.2x average length add a 'bond'
                if dist_mat[i, j] <= avg_bond_length * (1.0 + rel_tolerance) and (i, j) not in graph.edges:
                    graph.add_edge(i, j, pi=False, active=False)

    species.graph = graph
    set_graph_attributes(species)

    if not allow_invalid_valancies:
        remove_bonds_invalid_valancies(species)

    return None


def get_atom_ids_sorted_type(species):
    """
    Get a list of atom ids sorted by increasing atomic weight, useful for when a molecular graph depends on the order
    of atoms in what will be considered bonded

    Arguments:
        species (autode.species.Species):

    Returns:
        (list(int)):
    """
    return sorted(list(range(species.n_atoms)), key=lambda i: get_atomic_weight(atom_label=species.atoms[i].label))


def remove_bonds_invalid_valancies(species):
    """
    Remove invalid valencies for atoms that exceed their maximum valencies e.g. H should have
    no more than 1 'bond'

    Arguments:
        species (autode.species.Species):
    """

    for i in species.graph.nodes:

        max_valance = get_maximal_valance(atom_label=species.atoms[i].label)
        neighbours = list(species.graph.neighbors(i))

        if len(neighbours) <= max_valance:
            # All is well
            continue

        logger.warning(f'Atom {i} exceeds its maximal valence removing edges')

        # Get the atom indexes sorted by the closest to atom i
        closest_atoms = sorted(neighbours, key=lambda j: species.get_distance(i, j))

        # Delete all the bonds to atom(s) j that are above the maximal valance
        for j in closest_atoms[max_valance:]:
            species.graph.remove_edge(i, j)

    return None


def set_graph_attributes(species):
    """
    For a molecular species set the π bonds and stereocentres in the molecular graph.

    Arguments:
        species (autode.species.Species):
    """
    logger.info('Setting the π bonds in a species')

    for bond in species.graph.edges:
        atom_i, atom_j = bond

        if all([is_pi_atom(atom_label=species.atoms[atom].label, valency=species.graph.degree[atom]) for atom in bond]):
            # TODO fix this for alternating single and double bonds, currently all shown as pi
            species.graph.edges[atom_i, atom_j]['pi'] = True

    logger.info('Setting the stereocentres in a species')

    for bond in [edge for edge in species.graph.edges if species.graph.edges[edge]['pi'] is True]:
        if is_chiral_pi_bond(species, bond):
            species.graph.nodes[bond[0]]['stereo'] = True
            species.graph.nodes[bond[1]]['stereo'] = True

    for atom in range(species.n_atoms):
        if is_chiral_atom(species, atom):
            species.graph.nodes[atom]['stereo'] = True

    return None


def union(graphs):
    """Return the union of two graphs. The disjoint union is returned"""
    if len(graphs) == 0:
        return nx.Graph()

    return nx.disjoint_union_all(graphs)


def species_are_isomorphic(species1, species2):
    """
    Check if two complexes are isomorphic in at least one of their conformers

    Arguments:
        species1 (autode.species.Species):
        species2 (autode.species.Species):

    Returns:
        (bool)
    """
    logger.info(f'Checking if {species1.name} and {species2.name} are isomorphic')
    if species1.graph is None or species2.graph is None:
        raise NoMolecularGraph

    if is_isomorphic(species1.graph, species2.graph):
        return True

    if species1.conformers is None and species2.conformers is None:
        logger.warning('Cannot check for isomorphic species conformers')
        return False

    # Conformers don't necessarily have molecular graphs, so make them all
    logger.disabled = True

    for species in (species1, species2):
        if species.conformers is None:
            continue

        for conformer in species.conformers:
            make_graph(conformer)

    logger.disabled = False

    # Check on all the pairwise combinations of species conformers looking for an isomorphism
    conformers1 = species1.conformers if species1.conformers is not None else [species1]
    conformers2 = species2.conformers if species2.conformers is not None else [species2]

    for conformer1 in conformers1:
        for conformer2 in conformers2:

            if is_isomorphic(conformer1.graph, conformer2.graph):
                return True

    return False


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


def get_graph_no_active_edges(graph):
    """
    Get a molecular graph without the active edges

    Arguments:
        graph (np.Graph):
    """

    graph_no_ae = graph.copy()
    active_edges = [edge for edge in graph.edges if graph.edges[edge]['active'] is True]

    for (atom_i, atom_j) in active_edges:
        graph_no_ae.remove_edge(atom_i, atom_j)

    return graph_no_ae


def is_isomorphic(graph1, graph2, ignore_active_bonds=False, timeout=5):
    """Check whether two NX graphs are isomorphic. Contains a timeout because the gm.is_isomorphic() method
    occasionally gets stuck

    Arguments:
        graph1 (nx.Graph): graph 1
        graph2 (nx.Graph): graph 2

    Keywords Arguments:
        ignore_active_bonds (bool):
        timeout (float): Timeout in seconds

    Returns:
        (bool): if the graphs are isomorphic
    """

    if ignore_active_bonds:
        graph1, graph2 = get_graph_no_active_edges(graph1), get_graph_no_active_edges(graph2)

    if isomorphism.faster_could_be_isomorphic(graph1, graph2):
        graph_matcher = isomorphism.GraphMatcher(graph1, graph2,
                                                 node_match=isomorphism.categorical_node_match('atom_label', 'C'))

        manager = mp.Manager()
        res = manager.dict()
        p = mp.Process(target=gm_is_isomorphic, args=(graph_matcher, res))
        p.start()             # Start the process
        p.join(timeout)       # Wait until the timeout

        if p.is_alive():
            p.terminate()
            logger.error('NX graph matching hanging')
            return False

        return list(res.values())[0]
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


def reac_graph_to_prod_graph(reac_graph, bond_rearrang):
    """Makes the graph of the product from the reactant and the bond rearrang, so it has the indices of the reactant

    Arguments:
        reac_graph (nx.Graph): graph of the reactant
        bond_rearrang (autode.bond_rearrangement.BondRearrangement): the bond rearrang linking reacs and prods

    Returns:
        nx.Graph: graph of the product with each atom indexed as in the reactants
    """
    prod_graph = deepcopy(reac_graph)
    for fbond in bond_rearrang.fbonds:
        prod_graph.add_edge(*fbond)
    for bbond in bond_rearrang.bbonds:
        prod_graph.remove_edge(*bbond)
    return prod_graph


def get_separate_subgraphs(graph):
    """Find all the unconnected graphs in a graph

    Arguments:
        graph (nx.Graph): graph

    Returns:
        list: list of graphs separate graphs
    """
    return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]


def split_mol_across_bond(graph, bond):
    """Gets a list of atoms on either side of a bond

    Arguments:
        graph (nx.Graph): molecular graph
        bond (tuple): list of bonds to be split across

    """
    graph_copy = graph.copy()

    graph_copy.remove_edge(*bond)
    split_subgraphs = get_separate_subgraphs(graph_copy)

    if len(split_subgraphs) != 2:
        raise CannotSplitAcrossBond

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


def get_active_mol_graph(species, active_bonds):
    """
    Get a molecular graph that includes 'active edges' i.e. bonds that are either made or broken in the reaction

    Arguments:
        species (autode.species.Species):
        active_bonds: (list(tuple(int)))

    """
    logger.info('Getting molecular graph with active edges')
    active_graph = species.graph.copy()

    for bond in active_bonds:

        # The graph has both (i, j) and (j, i) edges such that the order is not important
        if bond in species.graph.edges:
            active_graph.edges[bond]['active'] = True

        else:
            active_graph.add_edge(*bond, pi=False, active=True)

    logger.info(f'Modified and added a total of {len(active_bonds)} bonds to the molecular graph')

    species.graph = active_graph
    set_graph_attributes(species)

    return None


def get_truncated_active_mol_graph(graph, active_bonds):
    """
    Generate a truncated graph of a graph that only contains the active bond atoms and their nearest neighbours

    Arguments:
        graph (nx.Graph):
        active_bonds (list(tuple(int)):
    """

    t_graph = nx.Graph()

    # Add all nodes that connect active bonds
    for bond in active_bonds:

        for atom_index in bond:
            if atom_index not in t_graph.nodes:
                t_graph.add_node(atom_index)

        t_graph.add_edge(*bond, active=True)

    # For every active atom add the nearest neighbours
    for atom_index in deepcopy(t_graph.nodes):
        neighbours = graph.neighbors(atom_index)

        # Add nodes and edges for all atoms and bonds to the neighbours that don't already exist in the graph
        for n_atom_index in neighbours:
            if n_atom_index not in t_graph.nodes:
                t_graph.add_node(n_atom_index)

            if (atom_index, n_atom_index) not in t_graph.edges:
                t_graph.add_edge(atom_index, n_atom_index)

    logger.info(f'Truncated graph generated. {t_graph.number_of_nodes()} nodes and {t_graph.number_of_edges()} edges')
    return t_graph


def is_chiral_pi_bond(species, bond):
    """Determine if a pi bond is chiral, by seeing if either atom has the same group bonded to it twice"""

    for i, atom in enumerate(bond):
        neighbours = list(species.graph.neighbors(atom))
        neighbours.remove(bond[1-i])

        if len(neighbours) != 2:
            return False

        graphs = []
        for neighbour in neighbours:
            graph = species.graph.copy()
            graph.remove_edge(atom, neighbour)
            split_subgraphs = get_separate_subgraphs(graph)
            graphs.append([subgraph for subgraph in split_subgraphs if neighbour in list(subgraph.nodes())][0])

        if is_isomorphic(graphs[0], graphs[1], ignore_active_bonds=True):
            return False

    return True


def is_chiral_atom(species, atom):
    """Determine if an atom is chiral, by seeing if any of the bonded groups are the same"""
    neighbours = list(species.graph.neighbors(atom))

    if len(neighbours) != 4:
        return False

    graphs = []
    for neighbour in neighbours:
        graph = species.graph.copy()
        graph.remove_edge(atom, neighbour)
        split_subgraphs = get_separate_subgraphs(graph)
        graphs.append([subgraph for subgraph in split_subgraphs if neighbour in list(subgraph.nodes())][0])

    for graph1, graph2 in itertools.combinations(graphs, 2):
        if is_isomorphic(graph1, graph2, ignore_active_bonds=True):
            return False

    return True
