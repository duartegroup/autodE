from .log import logger
from .transition_state import TS
from .atoms import valid_valances
from .bond_lengths import get_avg_bond_length
from .mol_graphs import is_isomorphic
from .optts import get_orca_optts_out
from .ORCAio import get_imag_frequencies_xyzs_energy_optts
from .pes_1d import get_xtb_ts_guess_1dpes_scan


def find_tss(reaction):
    """
    :param reaction: Reaction object
    :return:
    """
    logger.info('Finding TSs for a rearrangement reaction')
    tss = []

    reactant, product = reaction.reacs[0], reaction.prods[0]
    fbond_and_bbond_ids = get_forming_and_breaking_bonds(reactant, product)

    for fbond_ids, bbond_ids in fbond_and_bbond_ids.items():
        for ts_guess_func in [get_xtb_ts_guess_forming_bond]:
            logger.info('Guessing at a TS geometry')
            ts_guess_xyzs = ts_guess_func(reactant, fbond_ids)

            if ts_guess_xyzs is not None:
                logger.info('Found a TS guess geometry with ' + ts_guess_func.__name__)
                optts_name = ts_guess_func.__name__ + '_' + str(fbond_ids[0]) + str(fbond_ids[1]) + '_TS'
                orca_out_lines = get_orca_optts_out(ts_guess_xyzs, reactant.charge, reactant.mult,
                                                    reactant.solvent, name=optts_name,
                                                    bond_ids_to_add=[fbond_ids])
                imag_freqs, ts_xyzs, ts_energy = get_imag_frequencies_xyzs_energy_optts(orca_out_lines)

                if len(imag_freqs) == 1:
                    logger.info('Found TS with 1 imaginary frequency')
                    tss.append(TS(imag_freqs, ts_xyzs, ts_energy))
                else:
                    logger.warning('OptTS calculation returned {} imaginary frequencies'.format(len(imag_freqs)))

    return tss


def get_xtb_ts_guess_forming_bond(reactant, fbond_ids, n_steps=20):
    atom_i, atom_j = fbond_ids
    logger.info('Performing XTB relaxed PES scan along atoms {}, {}'.format(atom_i, atom_j))
    atom_labels = [xyz[0] for xyz in reactant.xyzs]
    curr_fbond_dist = reactant.distance_matrix[atom_i][atom_j]
    final_fbond_dist = get_avg_bond_length(atom_labels[atom_i], atom_labels[atom_j])
    return get_xtb_ts_guess_1dpes_scan(reactant, fbond_ids, curr_fbond_dist, final_fbond_dist, n_steps)


def get_forming_and_breaking_bonds(reactant, product):
    logger.info('Getting forming and breaking bonds')

    forming_and_breaking_bonds = {}

    possible_forming_bonds = get_possible_forming_bonds(reactant)
    possible_breaking_bonds = get_possible_breaking_bonds(possible_forming_bonds, reactant.graph)

    for forming_bond in possible_forming_bonds:
        for breaking_bond in possible_breaking_bonds:

            reac_graph_rearrangement = generate_rearranged_graph(reactant.graph, forming_bond, breaking_bond)
            if is_isomorphic(reac_graph_rearrangement, product.graph):
                forming_and_breaking_bonds[forming_bond] = breaking_bond

    return forming_and_breaking_bonds


def get_possible_breaking_bonds(possible_forming_bonds, mol_graph):
    logger.info('Getting possible breaking bonds')
    return [pair for pair in mol_graph.edges() if sorted(pair) not in possible_forming_bonds]

    # possible_breaking_bonds = []

    # for bond in possible_forming_bonds:
    #     atom_i, atom_j = bond
    #     atom_i_label, atom_i_valance = mol_graph.nodes[atom_i]['atom_label'], len(mol_graph.edges(atom_i))
    #     if atom_i_valance == valid_valances[atom_i_label][-1]:               # Max valance is last in the list
    #         possible_breaking_bonds.append(mol_graph.edges(atom_i))

    # return None


def get_non_maximal_valance_atoms(mol_graph):
    """
    From a networkx molecular graph determine which atoms are not maximally coordinated and could support another bond
    (given some other bonds being made/broken)

    :param mol_graph: Molecule object
    :return: (list) atom ids that are not maximally coordinated
    """
    non_maximal_valance_atom_ids = []

    for atom in list(mol_graph.nodes):
        valance = len(mol_graph.edges(atom))
        try:
            max_valance = valid_valances[mol_graph.nodes[atom]['atom_label']][-1]     # Max valance is last in the list
            if valance < max_valance:
                non_maximal_valance_atom_ids.append(atom)
        except KeyError:
            logger.warning('Couldn\'t find valance for {}'.format(atom['atom_label']))

    return non_maximal_valance_atom_ids


def get_possible_forming_bonds(reactant):
    logger.info('Getting possible forming bonds that don\'t exceed standard valances')
    non_max_val_atom_ids = get_non_maximal_valance_atoms(mol_graph=reactant.graph)
    possible_forming_bonds = [(i, j) for i in range(reactant.n_atoms) for j in non_max_val_atom_ids if i != j]
    possible_forming_bonds = strip_non_unique_pairs(possible_forming_bonds)

    return possible_forming_bonds


def strip_non_unique_pairs(bond_pair_list):
    return list(set([tuple(sorted(item)) for item in bond_pair_list]))


def generate_rearranged_graph(graph, forming_bond, breaking_bond):
    """
    Generate a rearranged graph by breaking one bond (edge) and forming another (edge)
    :param graph: (nx graph object)
    :param forming_bond: (tuple) Forming bond ids
    :param breaking_bond: (tuple) Breaking bond ids
    :return:
    """
    rearranged_graph = graph.copy()
    rearranged_graph.add_edge(*forming_bond)
    rearranged_graph.remove_edge(*breaking_bond)

    return rearranged_graph
