import networkx as nx
from autode.log import logger
from copy import deepcopy
from autode.bond_lengths import get_avg_bond_length


def add_core_pi_bonds(molecule, s_molecule, truncated_graph):
    """
    Add π bonds that are nearest neighbours to the current atoms in the truncated graph

    Arguments:
        molecule (autode.species.Species):
        s_molecule (autode.species.Species): Stripped molecule
        truncated_graph (nx.Graph):
    """
    logger.info('Adding π bonds to the truncated graph')

    curr_nodes = deepcopy(truncated_graph.nodes)

    while True:

        for bond in s_molecule.graph.edges:

            if s_molecule.graph.edges[bond]['pi'] is False or bond in truncated_graph.edges:
                continue

            # At least one of the atoms in the bond needs to be in the current structure
            if all(atom_index not in curr_nodes for atom_index in bond):
                continue

            truncated_graph.add_nodes_from([(i, molecule.graph.nodes[i]) for i in bond if i not in truncated_graph.nodes])
            truncated_graph.add_edges_from([(*bond, molecule.graph.edges[bond])])

        if truncated_graph.number_of_nodes() == len(curr_nodes):
            break

        else:
            curr_nodes = deepcopy(truncated_graph.nodes)

    return curr_nodes


def add_capping_atom(atom_index, n_atom_index, graph, s_molecule):
    """
    Add a capping atom, e.g.


              H
             /
     C_a---C_b - H   ->   C_a--H          where C_a is numbered atom_index, C_b is numbered n_atom_index
             \
              H

    Arguments:
        atom_index (int):
        n_atom_index (int):
        graph (nx.Graph): Current molecular graph of the stripped/truncated molecule
        s_molecule (autode.species.Species): Stripped molecule
    """

    graph.add_node(n_atom_index, atom_label='H', stereo=False)

    # Relabel the atom in the stripped molecule
    s_molecule.atoms[n_atom_index].label = 'H'

    # Shift the added capping hydrogen to the 'ideal' E-H bond length
    curr_dist = s_molecule.get_distance(atom_index, n_atom_index)
    ideal_dist = get_avg_bond_length(atom_i_label=s_molecule.atoms[atom_index].label, atom_j_label='H')
    shift_vec = s_molecule.atoms[n_atom_index].coord - s_molecule.atoms[atom_index].coord

    s_molecule.atoms[n_atom_index].translate(vec=(ideal_dist - curr_dist) * shift_vec / curr_dist)

    return None


def add_capping_atoms(molecule, s_molecule, truncated_graph, curr_nodes):
    """
    Add capping atoms to the graph, truncating over C-C single bonds where appropriate

    Arguments:
        molecule (autode.species.Species):
        s_molecule (autode.species.Species): Stripped molecule
        truncated_graph (nx.Graph):
        curr_nodes (list(int)):
    """

    truncated_atom_indexes = []

    while True:

        for i in curr_nodes:

            if i in truncated_atom_indexes:
                # Truncated atoms by definition do not have any neighbours that are not already in the graph
                continue

            for n_atom_index in s_molecule.graph.neighbors(i):

                if n_atom_index in curr_nodes:
                    continue

                n_neighbours = len(list(s_molecule.graph.neighbors(n_atom_index)))
                if s_molecule.atoms[n_atom_index].label == 'C' and n_neighbours == 4:
                    truncated_atom_indexes.append(n_atom_index)

                    add_capping_atom(i, n_atom_index, graph=truncated_graph, s_molecule=s_molecule)

                else:
                    truncated_graph.add_nodes_from([(n_atom_index, molecule.graph.nodes[n_atom_index])])

                truncated_graph.add_edges_from([(i, n_atom_index, molecule.graph.edges[(i, n_atom_index)])])

        if truncated_graph.number_of_nodes() == len(curr_nodes):
            # No nodes have been added on this iteration
            break

        else:
            curr_nodes = deepcopy(truncated_graph.nodes)

    return None


def strip_non_core_atoms(molecule, active_atoms):
    """

    Arguments:
         molecule (autode.species.Species):
         active_atoms (list(int)):
    """
    s_molecule = deepcopy(molecule)

    logger.info(f'Truncating {molecule.name} with {molecule.n_atoms} atoms around core atoms: {active_atoms}')
    truncated_graph = nx.Graph()

    # Add all the core active atoms to the graphs, their nearest neighbours and the bonds between them
    truncated_graph.add_nodes_from([(i, molecule.graph.nodes[i]) for i in active_atoms])

    for i in active_atoms:
        truncated_graph.add_nodes_from([(j, molecule.graph.nodes[j]) for j in molecule.graph.neighbors(i)])
        truncated_graph.add_edges_from([(i, j, molecule.graph.edges[(i, j)]) for j in molecule.graph.neighbors(i)])

    # Add all the π bonds that are associated with the core atoms, then close those those etc.
    curr_nodes = add_core_pi_bonds(molecule, s_molecule, truncated_graph=truncated_graph)

    # Swap all unsaturated carbons and the attached fragment for H
    logger.warning('Truncation is only implemented over C-C single bonds')
    add_capping_atoms(molecule, s_molecule, truncated_graph=truncated_graph, curr_nodes=curr_nodes)

    # Delete all atoms not in the truncated graph and reset the graph
    s_molecule.graph = truncated_graph

    s_molecule.set_atoms(atoms=[atom for i, atom in enumerate(s_molecule.atoms) if i in sorted(truncated_graph.nodes)])

    # Relabel the nodes so they correspond to the new set of atoms
    mapping = {node_label: i for i, node_label in enumerate(sorted(truncated_graph.nodes))}
    s_molecule.graph = nx.relabel_nodes(s_molecule.graph, mapping=mapping)

    logger.info(f'Truncated to {s_molecule.n_atoms} atoms')
    return s_molecule


def get_truncated_complex(complex_, bond_rearrangement):
    """
    From a truncated reactant complex

    Arguments:
        complex_ (autode.complex.ReactantComplex):
        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
    """

    truncated_complex = strip_non_core_atoms(molecule=complex_,
                                             active_atoms=bond_rearrangement.active_atoms)
    truncated_complex.name += '_truncated'

    return truncated_complex


def is_worth_truncating(reactant_complex, bond_rearrangement):
    """
    Evaluate whether it is worth truncating a complex

    Arguments:
        reactant_complex (autode.complex.ReactantComplex):
        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
    """
    truncated_complex = get_truncated_complex(reactant_complex, bond_rearrangement)

    if reactant_complex.n_atoms - truncated_complex.n_atoms < 10:
        logger.info('Truncated complex had 10 atoms or fewer than the full complex. Not truncating')
        return False

    logger.info('Complex is worth truncating')
    return True
