from .config import Config
from .log import logger
from .geom import get_breaking_bond_atom_id_dist_dict
from .pes_1d import get_orca_ts_guess_1dpes_scan
from .pes_1d import get_xtb_ts_guess_1dpes_scan
from .pes_2d import get_orca_ts_guess_2d_breaking_bonds
from .geom import get_valid_mappings_frags_to_whole_graph
from .optts import get_ts


def find_ts(reaction):
    """
    Find a transition state for a dissociative reaction i.e. reactant1 -> product1 + product2
    :param reaction:
    :return:
    """
    logger.info('Finding TS for a dissociation reaction')
    reactant = reaction.reacs[0]
    bbond_ids = find_breaking_bond_ids(reaction)

    transition_state = find_ts_breaking_bond(reactant, bbond_ids)

    if transition_state is None:
        logger.error('Could not find a suitable transition state')

    return transition_state


def find_ts_breaking_bond(reactant, bbonds):

    bbond_atom_ids_and_dists = get_breaking_bond_atom_id_dist_dict(reactant.xyzs, bbonds)

    for ts_guess_func in get_ts_guess_functions(bbonds):
        logger.info('Guessing at a TS geometry')
        ts_guess_xyzs = ts_guess_func(reactant, bbond_atom_ids_and_dists)

        if ts_guess_xyzs is not None:
            logger.info('Found a TS guess geometry with ' + ts_guess_func.__name__)
            for ts_opt_func in [get_ts]:
                transition_state = ts_opt_func(ts_guess_xyzs, reactant.charge, reactant.mult, reactant.solvent,
                                               ts_guess_func.__name__ + '_TS', bbonds)
                if transition_state:
                    return transition_state
    return None


def find_breaking_bond_ids(reaction):
    logger.info('Finding breaking bond(s) for a dissociation reaction')

    reactant, prod1, prod2 = reaction.reacs[0], reaction.prods[0], reaction.prods[1]

    valid_mappings = get_valid_mappings_frags_to_whole_graph(whole_graph=reactant.graph, frag1_graph=prod1.graph,
                                                             frag2_graph=prod2.graph)
    bbond_atom_ids_list = get_breaking_bond_ids(reactant.graph, valid_mappings)
    logger.info('Found *{}* breaking bonds'.format(len(bbond_atom_ids_list)))

    return bbond_atom_ids_list


def get_orca_ts_guess_coarse(reactant, bbond_atom_ids_and_dists):
    logger.info('Running a coarse PES scan with keywords set in Config')
    atom_ids, dist = list(bbond_atom_ids_and_dists.items())[0]
    return get_orca_ts_guess_1dpes_scan(reactant, atom_ids, dist, final_dist=dist+1.5,  n_steps=10,
                                        orca_keywords=Config.scan_keywords, name='default')


def get_orca_ts_guess_coarse_alt(reactant, bbond_atom_ids_and_dists):
    logger.info('Running a coarse PES scan at PBE0-D3BJ/de2-SVP')
    kws = ['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
    atom_ids, dist = list(bbond_atom_ids_and_dists.items())[0]
    return get_orca_ts_guess_1dpes_scan(reactant, atom_ids, dist, final_dist=dist+1.5, n_steps=10,
                                        orca_keywords=kws, name='alt')


def get_xtb_ts_guess_breaking_bond(reactant, bbond_atom_ids_and_dists):
    atom_ids, dist = list(bbond_atom_ids_and_dists.items())[0]
    return get_xtb_ts_guess_1dpes_scan(reactant, atom_ids, dist, final_dist=dist+1.5, n_steps=20)


def get_breaking_bond_ids(reactant_graph, valid_mappings):
    """
    For a list of valid mappings determine the breaking bonds (returned as a list of tuples, which are atoms ids
    defining the bond) these will have inter-fragment bonds..

    :param reactant_graph: networkX graph
    :param valid_mappings: (list(tuple(dict))) List of valid mappings
    :return:
    """

    breaking_bond_atom_ids_list = []
    for mapping_pair in valid_mappings:
        for frag1_atom_id in mapping_pair[0].keys():
            for frag2_atom_id in mapping_pair[1].keys():
                atom_ij = frag1_atom_id, frag2_atom_id
                if reactant_graph.has_edge(*atom_ij) and atom_ij not in breaking_bond_atom_ids_list:
                    breaking_bond_atom_ids_list.append(atom_ij)

    return breaking_bond_atom_ids_list


def get_ts_guess_functions(bbond_ids):
    """
    Get functions that will find TS guesses given the number of breaking bonds in the TS
    :param bbond_ids:
    :return:
    """

    if len(bbond_ids) == 1:
        return [get_orca_ts_guess_coarse, get_orca_ts_guess_coarse_alt]  # get_xtb_ts_guess_breaking_bond
    elif len(bbond_ids) == 2:
        return [get_orca_ts_guess_2d_breaking_bonds]                     # get_xtb_ts_guess_2d
    else:
        logger.critical('Can\'t yet handle >2 or 0 bonds changing')
        exit()
