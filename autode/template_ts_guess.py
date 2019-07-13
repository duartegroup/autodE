from .mol_graphs import get_mapping_ts_template
from .opt import get_orca_ts_guess_constrained_opt
from .templates import get_ts_templates
from .templates import template_matches
from .log import logger
from .config import Config
from .ts_guess import TSguess


def get_template_ts_guess(mol, active_bonds, reaction_class):
    """
    Get a transition state guess object by searching though the stored TS templates
    :param mol: (object) Molecule object
    :param active_bonds: (list(tuple)) List of active bonds in the TS, defined by atom IDs of the atom pair as a tuple
    :param reaction_class: (object) Reaction class (reactions.py)
    :return: TSguess object
    """
    logger.info('Getting TS guess from stored TS template')
    active_bonds_and_dists_ts = {}

    mol_graph = mol.get_active_mol_graph(active_bonds)                    # This will add edges so don't modify in place
    ts_guess_templates = get_ts_templates(reaction_class)

    for ts_template in ts_guess_templates:

        if template_matches(mol=mol, ts_template=ts_template, mol_graph=mol_graph):
            mapping = get_mapping_ts_template(larger_graph=mol_graph, smaller_graph=ts_template.graph)

            for active_bond in active_bonds:
                i, j = active_bond
                active_bonds_and_dists_ts[active_bond] = ts_template.graph.edges[mapping[i], mapping[j]]['weight']

            logger.info('Found a TS guess from a template')
            return get_orca_ts_guess_constrained_opt(mol, orca_keywords=Config.opt_keywords, name='ts_guess',
                                                     distance_constraints=active_bonds_and_dists_ts,
                                                     reaction_class=reaction_class)

    logger.info('Couldn\'t find a TS guess from a template')
    return TSguess(xyzs=None)
