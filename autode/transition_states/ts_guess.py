from typing import Optional
from autode.transition_states.base import TSbase
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.templates import template_matches
from autode.calculation import Calculation
from autode.config import Config
from autode.values import Distance
from autode.exceptions import CalculationException
from autode.log import logger
from autode.utils import work_in
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
                          method:     'autode.wrappers.base.ElectronicStructureMethod'):
    """
    Get a transition state guess object by searching though the stored TS
    templates

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.ReactantComplex):

        bond_rearr (autode.bond_rearrangement.BondRearrangement):

        product (autode.complex.ProductComplex):

        method (autode.wrappers.base.ElectronicStructureMethod):

        name (str):

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
        ts_guess = TSguess(atoms=reactant.atoms,
                           reactant=reactant,
                           product=product,
                           bond_rearr=bond_rearr)
        ts_guess.constraints.distance = active_bonds_and_dists_ts

        try:
            ts_guess.run_constrained_opt(name=name,
                                         method=method,
                                         keywords=method.keywords.opt)
            return ts_guess

        except CalculationException:
            logger.warning('Failed to run constrained optimisation on the TS')
            continue

    return None


class TSguess(TSbase):
    """Transition state guess"""

    @work_in('scan_to_template')
    def _lmethod_scan_to_point(self):
        """
        Run a set of constrained low-level optimisations from the current
        distances to the final set of constraints using a linear path with
        small distance increments.

        Raises:
            (autode.exceptions.CalculationException):
        """
        l_method = get_lmethod()

        # Final and current sets of constraints
        f_consts = self.constraints.distance
        c_consts = {bond: self.distance(*bond) for bond in f_consts.keys()}

        # Number of steps to use is 0.1 Å in the maximum distance delta
        max_delta = max(abs(f_consts[bond] - c_dist) for bond, c_dist in c_consts.items())
        n_steps = int(max_delta / Distance(0.1, units='ang'))

        if n_steps < 2:
            logger.info(f'No need to scan - only going to do {n_steps} steps')
            return

        for i in range(1, n_steps+1):

            consts = {}
            for bond, c_dist in c_consts.items():
                consts[bond] = c_dist + i * (f_consts[bond] - c_dist) / n_steps

            opt = Calculation(name=f'{self.name}_const_opt_ll_{i}',
                              molecule=self,
                              method=l_method,
                              keywords=l_method.keywords.low_opt,
                              n_cores=Config.n_cores,
                              distance_constraints=consts)

            # Can raise CalculationException
            self.optimise(calc=opt)

        return None

    def run_constrained_opt(self,
                            name:     str,
                            method:   Optional['autode.wrappers.base.ElectronicStructureMethod'] = None,
                            keywords: Optional['autode.wrappers.keywords.Keywords'] = None):
        """Get a TS guess from a constrained optimisation with the active atoms
        fixed at values defined in distance_consts

        Arguments:
            name (str):

            keywords (autode.wrappers.keywords.Keywords):

        Keyword Arguments:

            method (autode.wrappers.base.ElectronicStructureMethod): if
                   None then use the default method

            keywords (autode.wrappers.keywords.Keywords): If None then use
                     the default optimisation keywords

        Raises:
            (autode.exceptions.CalculationException):
        """
        logger.info('Running constrained optimisation on TS guess geometry')

        self._lmethod_scan_to_point()

        # Default to high-level regular optimisations
        if method is None:
            method = get_hmethod()
        if keywords is None:
            keywords = method.keywords.opt

        hl_const_opt = Calculation(name=f'{name}_constrained_opt',
                                   molecule=self,
                                   method=method,
                                   keywords=keywords,
                                   n_cores=Config.n_cores)

        self.optimise(calc=hl_const_opt)
        self.constraints.distance.clear()
        return None
