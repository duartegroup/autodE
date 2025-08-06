import itertools
import networkx as nx
import numpy as np

from copy import deepcopy
from networkx.algorithms import isomorphism
from typing import Optional, List, Tuple, TYPE_CHECKING
from autode.utils import timeout
import autode.exceptions as ex
from scipy.spatial import distance_matrix
from autode.atoms import Atom, metals
from autode.log import logger

if TYPE_CHECKING:
    from autode.values import Distance
    from autode.species.species import Species

_METAL_HAPTIC_THRESH = 1.10  # 10%


class MolecularGraph(nx.Graph):
    def __repr__(self):
        return (
            f"MolecularGraph(|E| = {self.number_of_edges()}, "
            f"|V| = {self.number_of_nodes()})"
        )

    @property
    def expected_planar_geometry(self) -> bool:
        """
        Is the 3D structure of the molecule corresponding to this graph
        expected to be planar?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """

        for node in self.nodes:
            n_neighbours = len(list(self.neighbors(node)))

            if n_neighbours < 4:
                # 1, 2 and 3-valent atoms must be planar
                continue

            if n_neighbours == 4 and self.nodes[node]["atom_label"] in metals:
                # Metals e.g. Rh can be square planar
                continue

            return False

        return True

    @property
    def eqm_bond_distance_matrix(self) -> np.ndarray:
        """
        An n_atoms x n_atoms matrix of ideal bond lengths. All non-bonded atoms
        will have zero ideal bond lengths

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): Matrix of bond lengths (Å)
        """
        logger.info("Getting ideal bond length matrix")

        n_atoms = self.number_of_nodes()
        matrix = np.zeros((n_atoms, n_atoms))

        for i, j in self.edges:
            r0 = float(self._covalent_radius(i) + self._covalent_radius(j))
            matrix[i, j] = matrix[j, i] = r0

        return matrix

    def _covalent_radius(self, i) -> "Distance":
        """Covalent radius of a node in the graph"""
        return Atom(self.nodes[i]["atom_label"]).covalent_radius.to("Å")

    def is_isomorphic_to(self, other: "MolecularGraph") -> bool:
        """Is this graph isomorphic to another?"""

        return is_isomorphic(self, other)

    @property
    def active_bonds(self) -> List[Tuple[int, int]]:
        """
        Extract the active bonds from the graph into a flat list of pairs
        of atom indices
        """
        return [
            (i, j) for (i, j) in self.edges if self.edges[(i, j)]["active"]
        ]

    def add_active_edge(self, u: int, v: int) -> None:
        """
        Add an 'active' edge between two atoms, where an active edge is one
        that is made or broken in a reaction
        """
        logger.info("Getting molecular graph with active edges")

        # The graph has both (i, j) and (j, i) edges thus order invariant
        if (u, v) in self.edges:
            self.edges[(u, v)]["active"] = True
        else:
            self.add_edge(u, v, pi=False, active=True)

        _set_graph_attributes(self)
        return None

    @property
    def node_matcher(self):
        """Default node matcher"""

        matcher = isomorphism.categorical_node_match(
            attr=["atom_label", "atom_class"], default=["C", None]
        )

        return matcher

    @property
    def is_connected(self) -> bool:
        """Is this graph fully connected (i.e. not separate parts)"""
        return nx.is_connected(self)

    def connected_components(self):
        """Generate the separate connected components"""
        return nx.connected_components(self)


def make_graph(
    species: "Species",
    rel_tolerance: float = 0.3,
    bond_list: Optional[List[tuple]] = None,
    allow_invalid_valancies: bool = False,
) -> None:
    """
    Make the molecular graph from the 'bonds' determined on a distance criteria
    or a SMILES parser object. All attributes default to false::

        Nodes attributes;
            (0) atom_label: Atomic symbol of this atom
            (1) stereo: Is this atom part of some stereochemistry e.g. R/S or
                        E/Z

        Edge attributes;
            (1) pi: Is this bond a pi bond. If it is then there should be no
                    rotation the bond axis in conformer generation
            (2) active: Is this bond being made/broken
                       (applies only to TransitionState objects)


    ---------------------------------------------------------------------------
    Arguments:
        species:

        rel_tolerance: Relative tolerance on what is considered a bond. E.g
                       0.3 means anything 1.3 x r0(i, j) is not 'bonded'.
                       Subject to the other restrictions on valency

        bond_list: Explicit bonds between atoms, overriding any attempt to
                   evaluate bonds

        allow_invalid_valancies: Should invalid atomic valencies be allowed?
                                 If false then e.g. a carbon atom with 5 atoms
                                 close enough to be bonded will have only 4 bonds

    Raises:
        NoAtomsInMolecule:
    """

    if species.n_atoms == 0:
        raise ex.NoAtomsInMolecule(
            "Could not build a molecular graph with no " "atoms"
        )

    logger.info("Generating molecular graph with NetworkX")

    graph = MolecularGraph()

    # Add the atoms to the graph all are initially assumed not to be
    # stereocenters
    for i, atom in enumerate(species.atoms):
        graph.add_node(
            i, atom_label=atom.label, stereo=False, atom_class=atom.atom_class
        )

    if bond_list is not None:
        logger.debug(
            f"Bonds have been specified. Adding {len(bond_list)} edges"
        )
        for i, j in bond_list:
            graph.add_edge(i, j, pi=False, active=False)

        species.graph = graph
        return None

    # Loop over the unique pairs of atoms and add 'bonds'
    coords = species.coordinates
    dist_mat = distance_matrix(coords, coords)

    # Enumerate atoms low->high atomic weight for consistent graph generation
    for i, _ in enumerate(
        sorted(species.atoms, key=lambda _atom: _atom.weight)
    ):
        # Iterate through the closest atoms to atom i
        for j in np.argsort(dist_mat[i]):
            if i == j:  # Don't bond atoms to themselves
                continue

            # Get r_avg for this X-Y bond e.g. C-C -> 1.5 Å
            avg_bond_length = species.atoms.eqm_bond_distance(i, j)

            # If the distance between atoms i and j are less or equal to
            # 1.25x average length add a 'bond'
            if (
                dist_mat[i, j] <= avg_bond_length * (1.0 + rel_tolerance)
                and (i, j) not in graph.edges
            ):
                graph.add_edge(i, j, pi=False, active=False)

    _set_graph_attributes(graph)
    species.graph = graph

    if not allow_invalid_valancies:
        remove_bonds_invalid_valancies(species)

    _sanitise_metal_bonds(species)
    return None


def _sanitise_metal_bonds(species):
    """
    Sometimes, covalent radii based bond detection adds extra bonds to
    metals in a haptic-type pattern (adjacent atoms to coordinating atoms
    are also bonded). These are removed by an approximate distance
    based criteria (NOT always perfect).

    Args:
        species: The species object
    """

    for i in species.graph.nodes:
        if not species.atoms[i].is_metal:
            continue

        to_remove = set()
        neighbours = list(species.graph.neighbors(i))
        haptic_groups: List[set] = []

        # Obtain sets of "haptic" atoms which are bonded to i and each other
        subset_graph = species.graph.subgraph(neighbours)
        haptic_groups = list(subset_graph.connected_components())

        for group in haptic_groups:
            if len(group) == 1:
                continue
            idxs_list = np.array(list(group))
            # bond length to metal scaled by sum of covalent radii
            r_i_x_scaled = [
                species.distance(i, x) / species.eqm_bond_distance(i, x)
                for x in idxs_list
            ]
            # any bond more than 10% weaker than the shortest is removed
            relative_ratios = np.array(r_i_x_scaled) / min(r_i_x_scaled)
            remove_idxs = idxs_list[
                np.where(relative_ratios >= _METAL_HAPTIC_THRESH)[0]
            ]
            to_remove.update(list(remove_idxs))

        if len(to_remove) != 0:
            logger.debug(
                f"Removing long 'haptic' bond(s) to metal {i}: {to_remove}"
            )
        for j in to_remove:
            species.graph.remove_edge(i, j)

    return None


def remove_bonds_invalid_valancies(species):
    """
    Remove invalid valencies for atoms that exceed their maximum valencies e.g.
    H should have no more than 1 'bond'. Does *NOT* consider coordinate bonds to
    metals.

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
    """

    for i in species.graph.nodes:
        if species.atoms[i].is_metal:
            continue

        max_valance = species.atoms[i].maximal_valance
        all_neighbours = list(species.graph.neighbors(i))
        non_m_neighbours = [
            j for j in all_neighbours if not species.atoms[j].is_metal
        ]

        if len(non_m_neighbours) <= max_valance:
            continue  # All is well

        logger.debug(f"Atom {i} exceeds its maximal valence removing edges")

        # Get the atom indexes sorted by the closest to atom i
        closest_atoms = sorted(
            non_m_neighbours, key=lambda k: species.distance(i, k)
        )

        # Delete all the bonds to atom(s) j that are above the maximal valance
        for j in closest_atoms[max_valance:]:
            species.graph.remove_edge(i, j)

    return None


def _set_graph_attributes(graph):
    """
    For a molecular species set the π bonds and stereocentres in the molecular
    graph.
    """
    logger.info("Setting graph attributes, inc. the π bonds")

    def is_idx_pi_atom(i):
        return Atom(graph.nodes[i]["atom_label"]).is_pi(
            valency=graph.degree[i]
        )

    for bond in graph.edges:
        atom_i, atom_j = bond

        if all(is_idx_pi_atom(i) for i in bond):
            graph.edges[atom_i, atom_j]["pi"] = True

    logger.info("Setting the stereocentres in a species")
    # List of atom indexes that are rings in the species
    rings = find_cycles(graph)

    for i, j in graph.edges:
        if graph.edges[(i, j)]["pi"] is False:
            continue

        if any(i in ring for ring in rings):
            # The ring should define the stereochemistry of this pi bond
            continue

        if _is_stereo_pi_bond(graph, bond=(i, j)):
            graph.nodes[i]["stereo"] = True
            graph.nodes[j]["stereo"] = True

    for i in graph.nodes:
        if _is_chiral_atom(graph, atom_index=int(i)):
            graph.nodes[i]["stereo"] = True

    return None


def union(graphs):
    """Return the union of two graphs. The disjoint union is returned"""
    if len(graphs) == 0:
        return MolecularGraph()

    return nx.disjoint_union_all(graphs)


def species_are_isomorphic(species1, species2):
    """
    Check if two complexes are isomorphic in at least one of their conformers

    ---------------------------------------------------------------------------
    Arguments:
        species1 (autode.species.Species):

        species2 (autode.species.Species):

    Returns:
        (bool):
    """
    logger.info(
        f"Checking if {species1.name} and {species2.name} are " f"isomorphic"
    )

    if species1.graph is None or species2.graph is None:
        raise ex.NoMolecularGraph

    if is_isomorphic(species1.graph, species2.graph):
        return True

    if species1.n_conformers == species2.n_conformers == 0:
        logger.warning("Cannot check for isomorphic species conformers")
        return False

    # Conformers don't necessarily have molecular graphs, so make them all
    logger.disabled = True

    for species in (species1, species2):
        if species.n_conformers == 0:
            continue

        for conformer in species.conformers:
            make_graph(conformer)

    logger.disabled = False

    # Check on all the pairwise combinations of species conformers looking for
    #  an isomorphism
    def conformers_or_self(_species):
        """If there are no conformers for this species return itself otherwise
        the list of conformers"""
        if _species.n_conformers == 0:
            return [_species]

        return _species.conformers

    # Check on all pairs of conformers between the two species
    for conformer1 in conformers_or_self(species1):
        for conformer2 in conformers_or_self(species2):
            if is_isomorphic(conformer1.graph, conformer2.graph):
                return True

    return False


def graph_matcher(graph1: MolecularGraph, graph2: MolecularGraph):
    """
    Generate a networkX graph matcher between two graphs, matching on atom
    types and active bonds

    ---------------------------------------------------------------------------
    Arguments:
        graph1 (nx.Graph):

        graph2 (nx.Graph):

    Returns:
        (nx.GraphMatcher)
    """
    # Match on active edges too, with the default being false
    edge_match = isomorphism.categorical_edge_match("active", False)

    gm = isomorphism.GraphMatcher(
        graph1, graph2, node_match=graph1.node_matcher, edge_match=edge_match
    )
    return gm


def is_subgraph_isomorphic(
    larger_graph: MolecularGraph, smaller_graph: MolecularGraph
):
    """
    Is the smaller graph subgraph isomorphic to the larger graph?

    ---------------------------------------------------------------------------
    Arguments:
        larger_graph (nx.Graph):

        smaller_graph (nx.Graph):

    Returns:
        (bool)
    """
    logger.info("Running subgraph isomorphism")

    gm = graph_matcher(larger_graph, smaller_graph)
    if gm.subgraph_is_isomorphic():
        return True

    return False


def get_mapping_ts_template(
    larger_graph: MolecularGraph, smaller_graph: MolecularGraph
):
    """
    Find the mapping for a graph onto a TS template (smaller). Can raise
    StopIteration with no match!

    ---------------------------------------------------------------------------
    Arguments:
        larger_graph (nx.Graph):

        smaller_graph (nx.Graph):

    Returns:
        (dict): Mapping
    """
    logger.info("Getting mapping of molecule onto the TS template")

    gm = graph_matcher(larger_graph, smaller_graph)

    return next(gm.match())


def get_mapping(graph1, graph2):
    """
    Get a sorted mapping of nodes between two graphs

    ---------------------------------------------------------------------------
    Arguments:
        graph1 (nx.Graph):

        graph2 (nx.Graph):

    Returns:
        (dict)
    """
    logger.info("Running isomorphism")

    node_match = isomorphism.categorical_node_match("atom_label", "C")
    gm = isomorphism.GraphMatcher(graph1, graph2, node_match=node_match)

    try:
        mapping = next(gm.match())
    except StopIteration:
        raise ex.NoMapping

    return {i: mapping[i] for i in sorted(mapping)}


def reorder_nodes(graph, mapping):
    """
    Reorder the nodes in a graph using a mapping. NetworkX uses the inverse
    mapping so the dict is swapped before the nodes are relabeled

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph):

        mapping (dict):

    Returns:
        (nx.Graph)
    """
    return nx.relabel_nodes(
        graph, mapping={u: v for v, u in mapping.items()}, copy=True
    )


def get_graph_no_active_edges(graph):
    """
    Get a molecular graph without the active edges

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph):

    Returns:
        (nx.Graph):
    """

    graph_no_ae = graph.copy()
    active_edges = [
        edge for edge in graph.edges if graph.edges[edge]["active"] is True
    ]

    for i, j in active_edges:
        graph_no_ae.remove_edge(i, j)

    return graph_no_ae


def get_graphs_ignoring_active_edges(graph1, graph2):
    """
    Remove any active edges that are in either graph1 or graph2 from both
    graphs

    ---------------------------------------------------------------------------
    Arguments:
        graph1 (nx.Graph):

        graph2 (nx.Graph):

    Returns:
        (tuple(nx.Graph))
    """
    g1, g2 = graph1.copy(), graph2.copy()

    # Iterate through the pairs removing any active edges from both ga and gb
    for ga, gb in [(g1, g2), (g2, g1)]:
        for edge in ga.edges:
            if ga.edges[edge]["active"] is False:
                continue

            i, j = edge
            ga.remove_edge(i, j)

            if (i, j) in gb.edges:
                gb.remove_edge(i, j)

    return g1, g2


def spectral_could_be_isomorphic(
    graph1: MolecularGraph, graph2: MolecularGraph
):
    """
    A fast check for whether two molecular graphs could be
    isomorphic based on comparison of the spectra of the
    weighted Laplacian matrices

    Args:
        graph1:
        graph2:

    Returns:
        (bool): If the graphs could be isomorphic
    """
    # if there are no bonds, then this check cannot be used
    if graph1.number_of_edges() < 1:
        return True

    # First check element agnostic Laplacian matrix (faster)
    if not np.allclose(
        nx.laplacian_spectrum(graph1), nx.laplacian_spectrum(graph2)
    ):
        return False

    # Laplacian matrix weighted by atomic number products
    # If atomic label is missing, simply return
    atom_numbers_1, atom_numbers_2 = [], []
    for _, label in graph1.nodes(data="atom_label", default=None):
        if label is None:
            return True
        atom_numbers_1.append(Atom(label).atomic_number)
    for _, label in graph2.nodes(data="atom_label", default=None):
        if label is None:
            return True
        atom_numbers_2.append(Atom(label).atomic_number)
    wt_laplace_1 = nx.linalg.laplacian_matrix(graph1).toarray() / np.outer(
        atom_numbers_1, atom_numbers_1
    )
    wt_laplace_2 = nx.linalg.laplacian_matrix(graph2).toarray() / np.outer(
        atom_numbers_2, atom_numbers_2
    )
    evs1 = np.linalg.eigvalsh(wt_laplace_1)
    evs2 = np.linalg.eigvalsh(wt_laplace_2)

    if not np.allclose(evs1, evs2):
        return False
    else:
        return True


@timeout(seconds=5, return_value=False)
def is_isomorphic(
    graph1: MolecularGraph,
    graph2: MolecularGraph,
    ignore_active_bonds: bool = False,
) -> bool:
    """Check whether two NX graphs are isomorphic. Contains a timeout because
    the gm.is_isomorphic() method occasionally gets stuck

    ---------------------------------------------------------------------------
    Arguments:
        graph1:

        graph2:

        ignore_active_bonds (bool):

    Returns:
        (bool): if the graphs are isomorphic
    """

    if ignore_active_bonds:
        graph1, graph2 = get_graphs_ignoring_active_edges(graph1, graph2)

    if not isomorphism.faster_could_be_isomorphic(graph1, graph2):
        return False

    if not spectral_could_be_isomorphic(graph1, graph2):
        return False

    # Always match on atom types
    node_matcher = graph1.node_matcher

    if ignore_active_bonds:
        gm = isomorphism.GraphMatcher(graph1, graph2, node_match=node_matcher)

    else:
        # Also match on edges
        edge_match = isomorphism.categorical_edge_match("active", False)
        gm = isomorphism.GraphMatcher(
            graph1, graph2, node_match=node_matcher, edge_match=edge_match
        )

    return gm.is_isomorphic()


def gm_is_isomorphic(gm, result):
    result[0] = gm.is_isomorphic()


def find_cycles(graph):
    """Finds all the cycles in a graph

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph): the molecular graph

    Returns:
        list(list): each list has the atoms in a cycle
    """
    return nx.cycle_basis(graph)


def connected_components(graph):
    """Connected sections of the nx.Graph"""
    return list(nx.connected_components(graph))


def reac_graph_to_prod_graph(reac_graph, bond_rearrang):
    """Makes the graph of the product from the reactant and the bond
    rearrangement, so it has the indices of the reactant

    ---------------------------------------------------------------------------
    Arguments:
        reac_graph (nx.Graph): Graph of the reactant
        bond_rearrang (autode.bond_rearrangement.BondRearrangement): The bond
                       rearrangement linking reactants and products

    Returns:
        (nx.Graph): Graph of the product with each atom indexed as in the
                    reactants
    """
    prod_graph = deepcopy(reac_graph)

    for fbond in bond_rearrang.fbonds:
        prod_graph.add_edge(*fbond)

    for bbond in bond_rearrang.bbonds:
        prod_graph.remove_edge(*bbond)

    return prod_graph


def get_separate_subgraphs(graph):
    """
    Find all the unconnected graphs in a graph

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph): graph

    Returns:
        list: list of graphs separate graphs
    """
    return [graph.subgraph(c).copy() for c in nx.connected_components(graph)]


def split_mol_across_bond(graph, bond):
    """Gets a list of atoms on either side of a bond. Should be separable
    into two graphs

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph): Molecular graph

        bond (tuple(int)): Bond to be split across e.g. (0, 1)

    Returns:
        (list(list(int))): List of atom indexes (as a list of integers)
    """
    graph_copy = graph.copy()

    graph_copy.remove_edge(*bond)
    split_subgraphs = get_separate_subgraphs(graph_copy)

    if len(split_subgraphs) != 2:
        raise ex.CannotSplitAcrossBond

    return [list(graph.nodes) for graph in split_subgraphs]


def get_bond_type_list(graph) -> dict[str, list]:
    """
    Finds the types (i.e CH) of all the bonds in a molecular graph

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph): Molecular graph

    Returns:
        bond_list_dict (dict): key = bond type, value = list of bonds of this
                               type
    """
    bond_list_dict: dict[str, list] = {}
    atom_types = set()

    for _, atom_label in graph.nodes.data("atom_label"):
        atom_types.add(atom_label)

    ordered_atom_labels = sorted(atom_types)

    for index, atom_label in enumerate(ordered_atom_labels):
        for i in range(index, len(ordered_atom_labels)):
            key = atom_label + ordered_atom_labels[i]
            bond_list_dict[key] = []

    for bond in graph.edges:
        atom_i_label = graph.nodes[bond[0]]["atom_label"]
        atom_j_label = graph.nodes[bond[1]]["atom_label"]
        key1, key2 = atom_i_label + atom_j_label, atom_j_label + atom_i_label

        if key1 in bond_list_dict.keys():
            bond_list_dict[key1].append(bond)
        elif key2 in bond_list_dict.keys():
            bond_list_dict[key2].append(bond)

    return bond_list_dict


def get_fbonds(graph, key):
    """
    Get all the possible forming bonds of a certain type

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph): graph object of a molecule

        key (str): string representing the bond type to be examined

    Returns:
        list: list of bonds that can be made of this type
    """
    possible_fbonds = []
    bonds = list(graph.edges)
    for i in graph.nodes:
        for j in graph.nodes:
            if i >= j:
                continue

            if not (i, j) in bonds and not (j, i) in bonds:
                bond = (i, j)
                label_i = graph.nodes[bond[0]]["atom_label"]
                label_j = graph.nodes[bond[1]]["atom_label"]

                key1, key2 = label_i + label_j, label_j + label_i

                if key1 == key or key2 == key:
                    possible_fbonds.append(bond)

    return possible_fbonds


def get_truncated_active_mol_graph(graph, active_bonds=None):
    """
    Generate a truncated graph of a graph that only contains the active bond
    atoms and their nearest neighbours

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph):

        active_bonds (list(tuple(int)):
    """

    if active_bonds is None:
        # Molecular graph may already define the active edges
        active_bonds = [
            pair for pair in graph.edges if graph.edges[pair]["active"]
        ]

    if len(active_bonds) == 0:
        raise ValueError(
            "Could not generate truncated active molecular "
            "graph with no active bonds"
        )

    t_graph = MolecularGraph()

    # Add all nodes that connect active bonds
    for bond in active_bonds:
        for idx in bond:
            if idx not in t_graph.nodes:
                label = graph.nodes[idx]["atom_label"]
                t_graph.add_node(idx, atom_label=label)

        t_graph.add_edge(*bond, active=True, pi=False)

    # For every active atom add the nearest neighbours
    for idx in deepcopy(t_graph.nodes):
        neighbours = graph.neighbors(idx)

        # Add nodes and edges for all atoms and bonds to the neighbours that
        # don't already exist in the graph
        for n_atom_index in neighbours:
            if n_atom_index not in t_graph.nodes:
                label = graph.nodes[n_atom_index]["atom_label"]
                t_graph.add_node(n_atom_index, atom_label=label)

            if (idx, n_atom_index) not in t_graph.edges:
                t_graph.add_edge(idx, n_atom_index, pi=False, active=False)

    logger.info(
        f"Truncated graph generated. {t_graph.number_of_nodes()} "
        f"nodes and {t_graph.number_of_edges()} edges"
    )
    return t_graph


def _is_stereo_pi_bond(graph, bond):
    """Determine if a pi bond is chiral, by seeing if either atom has the same
    group bonded to it twice"""

    for i, atom in enumerate(bond):
        neighbours = list(graph.neighbors(atom))
        neighbours.remove(bond[1 - i])

        if len(neighbours) != 2:
            return False

        graphs = []
        for neighbour in neighbours:
            graph = graph.copy()
            graph.remove_edge(atom, neighbour)
            split_subgraphs = get_separate_subgraphs(graph)
            graphs.append(
                [
                    subgraph
                    for subgraph in split_subgraphs
                    if neighbour in list(subgraph.nodes())
                ][0]
            )

        if is_isomorphic(graphs[0], graphs[1], ignore_active_bonds=True):
            return False

    return True


def _is_chiral_atom(graph, atom_index):
    """Determine if an atom is chiral, by seeing if any of the bonded groups
    are the same"""
    neighbours = list(graph.neighbors(atom_index))

    if len(neighbours) != 4:
        return False

    graphs = []
    for neighbour in neighbours:
        _graph = graph.copy()
        _graph.remove_edge(atom_index, neighbour)
        split_subgraphs = get_separate_subgraphs(_graph)
        graphs.append(
            [
                subgraph
                for subgraph in split_subgraphs
                if neighbour in list(subgraph.nodes())
            ][0]
        )

    for graph1, graph2 in itertools.combinations(graphs, 2):
        if is_isomorphic(graph1, graph2, ignore_active_bonds=True):
            return False

    return True
