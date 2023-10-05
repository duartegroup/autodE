from copy import deepcopy
import networkx as nx
from autode.config import Config
from autode.atoms import Atom
from autode.transition_states.ts_guess import has_matching_ts_templates
from autode.mol_graphs import MolecularGraph
from autode.log import logger


def add_core_pi_bonds(molecule, s_molecule, truncated_graph):
    """
    Add π bonds that are nearest neighbours to the current atoms in the
    truncated graph

    ---------------------------------------------------------------------------
    Arguments:
        molecule (autode.species.Species):

        s_molecule (autode.species.Species): Stripped molecule

        truncated_graph (nx.Graph):
    """
    logger.info("Adding π bonds to the truncated graph")

    curr_nodes = deepcopy(truncated_graph.nodes)

    while True:
        for bond in s_molecule.graph.edges:
            if (
                s_molecule.graph.edges[bond]["pi"] is False
                or bond in truncated_graph.edges
            ):
                continue

            # At least one of the atoms in the bond needs to be in the
            # current structure
            if all(atom_index not in curr_nodes for atom_index in bond):
                continue

            nodes = [
                (i, molecule.graph.nodes[i])
                for i in bond
                if i not in truncated_graph.nodes
            ]
            truncated_graph.add_nodes_from(nodes)

            truncated_graph.add_edges_from(
                [(*bond, molecule.graph.edges[bond])]
            )

        if truncated_graph.number_of_nodes() == len(curr_nodes):
            break

        else:
            curr_nodes = deepcopy(truncated_graph.nodes)

    return curr_nodes


def add_capping_atom(atom_index, n_atom_index, graph, s_molecule):
    r"""
    Add a capping atom. Example::

                  H
                 /
         C_a---C_b - H   ->   C_a--H          where C_a is numbered atom_index,
                \                             C_b is numbered n_atom_index
                 \
                  H

    ---------------------------------------------------------------------------
    Arguments:
        atom_index (int):

        n_atom_index (int):

        graph (nx.Graph): Current molecular graph of the stripped/truncated
                          molecule

        s_molecule (autode.species.Species): Stripped molecule
    """
    logger.info(
        f"Swapping saturated carbon {n_atom_index} next to atom "
        f"{atom_index} for hydrogen"
    )

    graph.add_node(n_atom_index, atom_label="H", stereo=False)

    # Relabel the atom in the stripped molecule
    s_molecule.atoms[n_atom_index].label = "H"

    # Shift the added capping hydrogen to the 'ideal' E-H bond length
    curr_dist = s_molecule.distance(atom_index, n_atom_index)
    ideal_dist = (
        s_molecule.atoms[atom_index].covalent_radius
        + Atom("H").covalent_radius
    )
    shift_vec = (
        s_molecule.atoms[n_atom_index].coord
        - s_molecule.atoms[atom_index].coord
    )
    shift_vec *= (ideal_dist - curr_dist) / curr_dist

    s_molecule.atoms[n_atom_index].translate(vec=shift_vec)

    return None


def add_capping_atoms(molecule, s_molecule, truncated_graph, curr_nodes):
    """
    Add capping atoms to the graph, truncating over C-C single bonds where
    appropriate

    ---------------------------------------------------------------------------
    Arguments:
        molecule (autode.species.Species):

        s_molecule (autode.species.Species): Stripped molecule

        truncated_graph (nx.Graph):

        curr_nodes (list(int)):
    """
    # Set of atom indexes (R) that have been replaced for H
    truncated_nodes = []

    while True:
        for i in curr_nodes:
            if i in truncated_nodes:
                # Truncated atoms by definition do not have any neighbours
                # that are not already in the graph
                continue

            for n_atom_index in s_molecule.graph.neighbors(i):
                if (
                    n_atom_index in curr_nodes
                    or n_atom_index in truncated_nodes
                ):
                    continue

                n_neighbours = len(
                    list(s_molecule.graph.neighbors(n_atom_index))
                )

                # Three conditions that must be met for the n_atom_index -> H
                if (
                    s_molecule.atoms[n_atom_index].label == "C"
                    and n_neighbours == 4
                ):
                    truncated_nodes.append(n_atom_index)

                    add_capping_atom(
                        i,
                        n_atom_index,
                        graph=truncated_graph,
                        s_molecule=s_molecule,
                    )

                else:
                    truncated_graph.add_nodes_from(
                        [(n_atom_index, molecule.graph.nodes[n_atom_index])]
                    )

                truncated_graph.add_edges_from(
                    [
                        (
                            i,
                            n_atom_index,
                            molecule.graph.edges[(i, n_atom_index)],
                        )
                    ]
                )

        if truncated_graph.number_of_nodes() == len(curr_nodes):
            # No nodes have been added on this iteration
            break

        else:
            curr_nodes = deepcopy(truncated_graph.nodes)

    return None


def add_remaining_bonds(truncated_graph, full_graph):
    """Truncation by adding atoms and their nearest neighbours may miss bonds
    between sections that aren't connected initially, so add them"""

    for i, j in full_graph.edges:
        if i not in truncated_graph.nodes:
            continue
        # At least j in the graph

        if j not in truncated_graph.nodes:
            continue

        # i and j are in the graph
        if (i, j) in truncated_graph.edges:
            continue

        # Don't alter bonding if the atom has changed e.g. C -> H
        if any(
            truncated_graph.nodes[k]["atom_label"]
            != full_graph.nodes[k]["atom_label"]
            for k in (i, j)
        ):
            continue

        # an edge doesn't exist between atoms i ang j - make it
        truncated_graph.add_edge(i, j, pi=False, active=False)

    return None


def add_remaining_atoms(truncated_graph, full_graph, s_molecule):
    """Truncation can lead to a split across a C-C bond in a ring where one
    of the carbons is no longer has 4 nearest neighbours"""

    for i in deepcopy(truncated_graph.nodes):
        # No modification needed if the valency of this atom is retained
        n_truncated_neighbours = len(list(truncated_graph.neighbors(i)))
        n_full_neighbours = len(list(full_graph.neighbors(i)))

        if n_truncated_neighbours == n_full_neighbours:
            continue

        # Only consider non-swapped atoms e.g. not where C -> H
        if (
            truncated_graph.nodes[i]["atom_label"]
            != full_graph.nodes[i]["atom_label"]
        ):
            continue

        logger.warning(f"Atom {i} changed valency in truncation")
        for n in nx.neighbors(full_graph, i):
            if (i, n) in truncated_graph.edges:
                continue

            # Missing atom n from the truncated graph - probably truncated
            # X -> H but was also bonded to another atom also in the truncated
            # graph.
            x, y, z = s_molecule.atoms[n].coord
            s_molecule.atoms.append(Atom(atomic_symbol="Og", x=x, y=y, z=z))

            # Add the capping H atom in place of the X atom just added
            # will be the last atom index, if it's just been added
            add_capping_atom(
                atom_index=i,
                n_atom_index=len(s_molecule.atoms) - 1,
                graph=truncated_graph,
                s_molecule=s_molecule,
            )

            # Also add the edge between the added atom and the one that changed
            # valency
            truncated_graph.add_edge(
                i, len(s_molecule.atoms) - 1, pi=False, active=False
            )

        logger.info(
            f"New valency is {len(list(truncated_graph.neighbors(i)))}"
        )

    return None


def get_truncated_species(species, bond_rearrangement):
    """
    From a truncated species by removing non core atoms and adding
    capping atoms where appropriate

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):

    Returns:
        (autode.complex.ReactantComplex)
    """

    active_atoms = bond_rearrangement.active_atoms
    t_species = species.new_species(name=f"{species.name}_truncated")

    logger.info(
        f"Truncating {species.name} with {species.n_atoms} atoms "
        f"around core atoms: {active_atoms}"
    )
    t_graph = MolecularGraph()

    # Add all the core active atoms to the graphs, their nearest neighbours
    # and the bonds between them
    t_graph.add_nodes_from([(i, species.graph.nodes[i]) for i in active_atoms])

    for i in active_atoms:
        t_graph.add_nodes_from(
            [(j, species.graph.nodes[j]) for j in species.graph.neighbors(i)]
        )
        t_graph.add_edges_from(
            [
                (i, j, species.graph.edges[(i, j)])
                for j in species.graph.neighbors(i)
            ]
        )

    # Add all the π bonds that are associated with the core atoms, then close
    # those those etc.
    curr_nodes = add_core_pi_bonds(species, t_species, truncated_graph=t_graph)

    # Swap all saturated carbons and the attached fragment for H
    logger.warning("Truncation is only implemented over C-X single bonds")
    add_capping_atoms(
        species, t_species, truncated_graph=t_graph, curr_nodes=curr_nodes
    )

    add_remaining_bonds(t_graph, full_graph=species.graph)
    add_remaining_atoms(
        t_graph, full_graph=species.graph, s_molecule=t_species
    )

    # Delete all atoms not in the truncated graph and reset the graph
    t_species.graph = t_graph
    t_species.atoms = [
        atom
        for i, atom in enumerate(t_species.atoms)
        if i in sorted(t_graph.nodes)
    ]

    # Relabel the nodes so they correspond to the new set of atoms
    mapping = {
        node_label: i for i, node_label in enumerate(sorted(t_graph.nodes))
    }

    t_species.graph = nx.relabel_nodes(t_species.graph, mapping=mapping)

    logger.info(f"Truncated to {t_species.n_atoms} atoms")
    return t_species


def is_worth_truncating(reactant_complex, bond_rearrangement):
    """
    Evaluate whether it is worth truncating a complex

    ---------------------------------------------------------------------------
    Arguments:
        reactant_complex (autode.complex.ReactantComplex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
    """
    if has_matching_ts_templates(reactant_complex, bond_rearrangement):
        logger.info(
            "Not truncating a reactant (complex) that has a saved " "template"
        )
        return False

    truncated_complex = get_truncated_species(
        reactant_complex, bond_rearrangement
    )

    n_removed_atoms = reactant_complex.n_atoms - truncated_complex.n_atoms

    if n_removed_atoms < Config.min_num_atom_removed_in_truncation:
        logger.info(
            f"Truncated complex only had {n_removed_atoms} atoms "
            f"fewer than the full complex. Not truncating"
        )
        return False

    logger.info("Complex is worth truncating")
    return True
