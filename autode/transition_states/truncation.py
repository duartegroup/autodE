import networkx as nx
from autode.log import logger
from copy import deepcopy
from autode.conformers.conf_gen import get_simanl_atoms
from autode.complex import ReactantComplex, ProductComplex
from autode.molecule import Reactant, Product


def strip_non_core_atoms(molecule, active_atoms):
    """

    Arguments:
         molecule (autode.molecule.Molecule):
         active_atoms (list(int)):
    """
    logger.info(f'Truncating {molecule.name} with {molecule.n_atoms} atoms around core atoms: {active_atoms}')
    truncated_graph = nx.Graph()

    # Add all the core active atoms to the graphs, their nearest neighbours and the bonds between them
    for atom_index in active_atoms:
        truncated_graph.add_node(atom_index, atom_label=molecule.atoms[atom_index].label)

        for n_atom_index in molecule.graph.neighbors(atom_index):
            truncated_graph.add_node(n_atom_index, atom_label=molecule.atoms[atom_index].label)
            truncated_graph.add_edge(atom_index, n_atom_index)

    curr_nodes = deepcopy(truncated_graph.nodes)

    # Add all the π bonds that are associated with the core atoms, then close those those etc.
    while True:

        for bond in molecule.graph.edges:

            if not (molecule.graph.edges[bond]['pi'] is True and bond not in truncated_graph.edges):
                continue

            # At least one of the atoms in the bond needs to be in the
            if all(atom_index not in curr_nodes for atom_index in bond):
                continue

            for atom_index in bond:
                if atom_index not in truncated_graph.nodes:
                    truncated_graph.add_node(atom_index, atom_label=molecule.atoms[atom_index].label)

            truncated_graph.add_edge(*bond)

        if truncated_graph.number_of_nodes() == len(curr_nodes):
            break

        else:
            curr_nodes = deepcopy(truncated_graph.nodes)

    # Swap all unsaturated carbons and the attached fragment for H
    logger.warning('Truncation is only implemented over C-C single bonds')
    truncated_atoms = []

    while True:

        for atom_index in curr_nodes:

            if atom_index in truncated_atoms:
                # Truncated atoms by definition do not have any neighbours that are not already in the graph
                continue

            for n_atom_index in molecule.graph.neighbors(atom_index):

                if n_atom_index in curr_nodes:
                    continue

                if molecule.atoms[n_atom_index].label == 'C' and len(list(molecule.graph.neighbors(n_atom_index))) == 4:
                    truncated_graph.add_node(n_atom_index, atom_label='H')
                    truncated_graph.add_edge(atom_index, n_atom_index)
                    truncated_atoms.append( n_atom_index)

                else:
                    truncated_graph.add_node(n_atom_index, atom_label=molecule.atoms[n_atom_index].label)
                    truncated_graph.add_edge(atom_index, n_atom_index)

        if truncated_graph.number_of_nodes() == len(curr_nodes):
            # No nodes have been added on this iteration
            break

        else:
            curr_nodes = deepcopy(truncated_graph.nodes)

    # Delete all atoms not in the truncated graph and reset the graph
    molecule.graph = truncated_graph

    # Reset the atom labels as some may have changed
    for atom_index in truncated_graph.nodes:
        molecule.atoms[atom_index].label = truncated_graph.nodes[atom_index]['atom_label']

    molecule.set_atoms(atoms=[atom for i, atom in enumerate(molecule.atoms) if i in truncated_graph.nodes])

    logger.info(f'Truncated to {molecule.n_atoms} atoms')
    return molecule


def get_truncated_rcomplex(reactant_complex, bond_rearrangement):
    """
    From a truncated reactant complex

    Arguments:
        reactant_complex (autode.complex.ReactantComplex):
        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
    """
    truncated_molecules = []

    # Truncate each component in the reactant complex
    for molecule in reactant_complex.molecules:
        molecule = strip_non_core_atoms(molecule=deepcopy(molecule), active_atoms=bond_rearrangement.active_atoms)

        # Regenerate the 3D structure from the new graph
        atoms = get_simanl_atoms(species=molecule)

        # Form a new reactant using the same attributes as the original
        truncated_reactant = Reactant(name=molecule.name, atoms=atoms, charge=molecule.charge, mult=molecule.mult)
        truncated_reactant.solvent = molecule.solvent

        truncated_molecules.append(truncated_reactant)

    return ReactantComplex(*truncated_molecules)


def get_truncated_pcomplex(complex_graph, bond_rearrangement):


    return None


def is_worth_truncating(reactant_complex, bond_rearrangement):
    """
    Evaluate whether it is worth truncating a complex

    Arguments:
        reactant_complex (autode.complex.ReactantComplex):
        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
    """
    truncated_complex = get_truncated_rcomplex(reactant_complex, bond_rearrangement)

    if reactant_complex.n_atoms - truncated_complex.n_atoms < 5:
        logger.info('Truncated complex had 5 atoms or fewer than the full complex. Not truncating')
        return False

    logger.info('Complex is worth truncating')
    return True
