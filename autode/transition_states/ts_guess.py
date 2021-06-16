from autode.transition_states.base import TSbase
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.templates import template_matches
from autode.calculation import Calculation
from autode.config import Config
from autode.exceptions import CalculationException
from autode.log import logger
from autode.values import Distance
from autode.methods import get_lmethod, get_hmethod
from autode.mol_graphs import (get_mapping_ts_template,
                               get_truncated_active_mol_graph)


def has_matching_ts_templates(reactant:   'autode.species.ReactantComplex',
                              bond_rearr: 'autode.bond_rearrangement.BondRearrangement'):
    """
    See if there are any templates suitable to get a TS guess from a template

    Arguments:
        reactant (autode.complex.ReactantComplex):
        bond_rearr (autode.bond_rearrangement.BondRearrangement):

    Returns:
        (bool):
    """

    mol_graph = get_truncated_active_mol_graph(graph=reactant.graph,
                                               active_bonds=bond_rearr.all)
    ts_guess_templates = get_ts_templates()

    for ts_template in ts_guess_templates:

        if template_matches(reactant=reactant, ts_template=ts_template,
                            truncated_graph=mol_graph):
            return True

    return False


def get_template_ts_guess(reactant:   'autode.species.ReactantComplex',
                          product:    'autode.species.ProductComplex',
                          bond_rearr: 'autode.bond_rearrangement.BondRearrangement',
                          name:       str,
                          method:     'autode.wrappers.base.ElectronicStructureMethod',
                          dist_thresh: Distance = Distance(4.0, units='Å')):
    """Get a transition state guess object by searching though the stored TS
    templates

    Arguments:
        reactant (autode.complex.ReactantComplex):
        bond_rearr (autode.bond_rearrangement.BondRearrangement):
        product (autode.complex.ProductComplex):
        method (autode.wrappers.base.ElectronicStructureMethod):
        name (str):
        keywords (list(str)): Keywords to use for the ElectronicStructureMethod

    Keyword Arguments:
        dist_thresh (float): distance above which a constrained optimisation
                             probably won't work due to the initial geometry
                             being too far away from the ideal

    Returns:
        (autode.transition_states.ts_guess.TSguess):
    """
    logger.info('Getting TS guess from stored TS template')
    active_bonds_and_dists_ts = {}

    # This will add edges so don't modify in place
    mol_graph = get_truncated_active_mol_graph(graph=reactant.graph,
                                               active_bonds=bond_rearr.all)
    ts_guess_templates = get_ts_templates()

    for ts_template in ts_guess_templates:

        if not template_matches(reactant=reactant,
                                ts_template=ts_template,
                                truncated_graph=mol_graph):
            continue

        # Get the mapping from the matching template
        mapping = get_mapping_ts_template(larger_graph=mol_graph,
                                          smaller_graph=ts_template.graph)

        for active_bond in bond_rearr.all:
            i, j = active_bond
            logger.info(f'Mapping active bond {i}-{j}')

            try:
                dist = ts_template.graph.edges[mapping[i], mapping[j]]['distance']
                active_bonds_and_dists_ts[active_bond] = dist

            except KeyError:
                logger.warning(f'Couldn\'t find a mapping for bond {i}-{j}')

        if len(active_bonds_and_dists_ts) != len(bond_rearr.all):
            continue

        logger.info('Found a TS guess from a template')
        if any([reactant.distance(*bond) > dist_thresh for bond in bond_rearr.all]):
            logger.info(f'TS template has => 1 active bond distance larger '
                        f'than {dist_thresh}. Passing')
            continue

        ts_guess = TSguess(atoms=reactant.atoms,
                           reactant=reactant,
                           product=product,
                           bond_rearr=bond_rearr)

        try:
            ts_guess.run_constrained_opt(name=name,
                                         distance_consts=active_bonds_and_dists_ts,
                                         keywords=method.keywords.opt)
            return ts_guess

        except CalculationException:
            logger.warning('Failed top run constrained optimisation on the TS')
            continue

    return None


class TSguess(TSbase):
    """Transition state guess"""

    def run_constrained_opt(self,
                            name :           str,
                            distance_consts: dict,
                            keywords:        'autode.wrappers.keywords.Keywords'):
        """Get a TS guess from a constrained optimisation with the active atoms
        fixed at values defined in distance_consts

        Arguments:
            keywords (autode.wrappers.keywords.Keywords):
            name (str):
            distance_consts (dict): Distance constraints keyed with a tuple of
                                    atom indexes and value of the distance

        Raises:
            (autode.exceptions.CalculationException)
        """
        logger.info('Getting TS guess from constrained optimisation')

        ts_guess_const = self.copy()

        # Run a low level constrained optimisation first to prevent slow
        # high-level optimisation for a TS that is far from the current
        # geometry
        l_method = get_lmethod()
        ll_const_opt = Calculation(name=f'{name}_constrained_opt_ll',
                                   molecule=ts_guess_const,
                                   method=l_method,
                                   keywords=l_method.keywords.low_opt,
                                   n_cores=Config.n_cores,
                                   distance_constraints=distance_consts)

        # Try and set the atoms, but continue if they're not found as hopefully
        # the high-level method will be fine(?)
        try:
            ts_guess_const.optimise(calc=ll_const_opt)

        except CalculationException:
            logger.error('Failed to optimise with the low level method')

        hl_const_opt = Calculation(name=f'{name}_constrained_opt',
                                   molecule=ts_guess_const,
                                   method=get_hmethod(),
                                   keywords=keywords,
                                   n_cores=Config.n_cores,
                                   distance_constraints=distance_consts)

        ts_guess_const.optimise(calc=hl_const_opt)

        self.name = f'ts_guess_{name}'
        return
