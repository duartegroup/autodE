import base64
import hashlib
from copy import deepcopy
from autode.config import Config
from autode.solvent.solvents import get_solvent
from autode.transition_states.locate_tss import find_tss
from autode.exceptions import UnbalancedReaction
from autode.exceptions import SolventsDontMatch
from autode.log import logger
from autode.methods import get_hmethod
from autode.species.complex import get_complexes
from autode.species.molecule import Product
from autode.species.molecule import Reactant
from autode.plotting import plot_reaction_profile
from autode.units import KcalMol
from autode.utils import work_in
from autode.reactions import reaction_types


class Reaction:

    def __str__(self):
        """Return a very short 6 character hash of the reaction, not guaranteed
         to be unique"""

        name = (f'{self.name}_{"+".join([r.name for r in self.reacs])}--'
                f'{"+".join([p.name for p in self.prods])}')

        if self.solvent is not None:
            name += f'_{self.solvent.name}'

        hasher = hashlib.sha1(name.encode()).digest()
        return base64.urlsafe_b64encode(hasher).decode()[:6]

    def _check_balance(self):
        """Check that the number of atoms and charge balances between reactants
         and products. If they don't exit
        immediately
        """
        def total(molecules, attr):
            return sum([getattr(m, attr) for m in molecules])

        if total(self.reacs, 'n_atoms') != total(self.prods, 'n_atoms'):
            logger.critical('Number of atoms doesn\'t balance')
            raise UnbalancedReaction

        if total(self.reacs, 'charge') != total(self.prods, 'charge'):
            logger.critical('Charge doesn\'t balance')
            raise UnbalancedReaction

        self.charge = total(self.reacs, 'charge')
        return None

    def _check_solvent(self):
        """Check that all the solvents are the same for reactants and products
        """
        molecules = self.reacs + self.prods

        if self.solvent is None:
            if all([mol.solvent is None for mol in molecules]):
                logger.info('Reaction is in the gas phase')
                return

            elif all([mol.solvent is not None for mol in molecules]):
                if not all([mol.solvent == self.reacs[0].solvent for mol in molecules]):
                    logger.critical('Solvents in reactants and products '
                                    'don\'t match')
                    raise SolventsDontMatch

                else:
                    logger.info(f'Setting the reaction solvent to '
                                f'{self.reacs[0].solvent}')
                    self.solvent = self.reacs[0].solvent

            else:
                print([mol.solvent for mol in molecules])
                print([mol.name for mol in molecules])

                raise SolventsDontMatch

        if self.solvent is not None:
            logger.info(f'Setting solvent to {self.solvent.name} for all '
                        f'molecules in the reaction')

            for mol in self.reacs + self.prods:
                mol.solvent = self.solvent

        logger.info(f'Set the solvent of all species in the reaction to '
                    f'{self.solvent.name}')
        return None

    def switch_reactants_products(self):
        """Addition reactions are hard to find the TSs for, so swap reactants
        and products and classify as dissociation. Likewise for reactions wher
        the change in the number of bonds is negative
        """
        logger.info('Swapping reactants and products')

        self.prods, self.reacs = self.reacs, self.prods
        self.product, self.reactant = self.reactant, self.product
        return None

    def calc_delta_e(self):
        """Calculate the ∆Er of a reaction defined as
        ∆E = E(products) - E(reactants)

        Returns:
            (float): Energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')

        if any(mol.energy is None for mol in self.reacs + self.prods):
            logger.error('Cannot calculate ∆Er. At least one required energy '
                         'was None')
            return None

        return (sum([p.energy for p in self.prods]) -
                sum([r.energy for r in self.reacs]))

    def calc_delta_e_ddagger(self):
        """Calculate the ∆E‡ of a reaction defined as
         ∆E = E(ts) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        if self.ts is None:
            logger.error('No TS, cannot calculate ∆E‡')
            return None

        if self.ts.energy is None or any(r.energy is None for r in self.reacs):
            logger.error('TS or reactants had no energy, cannot calculate ∆E‡')
            return None

        return self.ts.energy - sum([r.energy for r in self.reacs])

    def find_lowest_energy_ts(self):
        """From all the transition state objects in Reaction.pes1d choose the
        lowest energy if there is more than one otherwise return the single
        transtion state or None if there no TS objects.
        """

        if self.tss is None:
            logger.error('Could not find a transition state')
            return None

        elif len(self.tss) > 1:
            logger.info('Found more than 1 TS. Choosing the lowest energy')
            min_ts_energy = min([ts.energy for ts in self.tss])
            return [ts for ts in self.tss if ts.energy == min_ts_energy][0]

        else:
            return self.tss[0]

    def find_lowest_energy_conformers(self):
        """Try and locate the lowest energy conformation using simulated
        annealing, then optimise them with xtb, then optimise the unique
        (defined by an energy cut-off) conformers with an electronic structure
        method"""

        h_method = get_hmethod() if Config.hmethod_conformers else None
        for mol in self.reacs + self.prods:
            # .find_lowest_energy_conformer works in conformers/
            mol.find_lowest_energy_conformer(hmethod=h_method)

        return None

    @work_in('reactants_and_products')
    def optimise_reacs_prods(self):
        """Perform a geometry optimisation on all the reactants and products
        using the method"""
        h_method = get_hmethod()
        logger.info(f'Optimising reactants and products with {h_method.name}')

        for mol in self.reacs + self.prods:
            mol.optimise(h_method)

        return None

    @work_in('complexes')
    def find_complexes(self):
        self.reactant, self.product = get_complexes(reaction=self)
        return None

    @work_in('complexes')
    def calculate_complexes(self):
        """Find the lowest energy conformers of reactant and product complexes
        using optimisation and single points"""
        h_method = get_hmethod()
        conf_hmethod = h_method if Config.hmethod_conformers else None

        for species in [self.reactant, self.product]:
            species.find_lowest_energy_conformer(hmethod=conf_hmethod)
            species.optimise(method=h_method)
            species.single_point(method=h_method)

        return None

    @work_in('transition_states')
    def locate_transition_state(self):

        if self.reactant is None and self.product is None:
            logger.warning('Reactant & product complexes are None- generating')
            self.find_complexes()

        # If there are more bonds in the product e.g. an addition reaction then
        # switch as the TS is then easier to find
        if (sum(p.graph.number_of_edges() for p in self.prods)
                > sum(r.graph.number_of_edges() for r in self.reacs)):

            self.switch_reactants_products()
            self.tss = find_tss(self)
            self.switch_reactants_products()

        else:
            self.tss = find_tss(self)

        self.ts = self.find_lowest_energy_ts()
        return None

    @work_in('transition_states')
    def find_lowest_energy_ts_conformer(self):
        """Find the lowest energy conformer of the transition state"""
        if self.ts is None:
            logger.error('No transition state to evaluate the conformer of')
            return None

        else:
            return self.ts.find_lowest_energy_ts_conformer()

    @work_in('single_points')
    def calculate_single_points(self):
        """Perform a single point energy evaluations on all the reactants and
        products using the hmethod"""
        h_method = get_hmethod()
        logger.info(f'Calculating single points with {h_method.name}')

        for mol in self.reacs + self.prods + [self.ts]:
            if mol is not None:
                mol.single_point(h_method)

        return None

    def _plot_reaction_profile_with_complexes(self, units):

        @work_in(self.name)
        def calculate_complexes():
            return self.calculate_complexes()

        calculate_complexes()

        reactions_wc = []

        # If the reactant complex contains more than one molecule then
        # make a reaction that is separated reactants -> reactant complex
        if len(self.reacs) > 1:
            reactant_complex = deepcopy(self.reactant)
            reactant_complex.__class__ = Product
            reactions_wc.append(Reaction(*self.reacs, reactant_complex,
                                         name='reactant_complex'))

        # The elementary reaction is then
        # reactant complex -> product complex
        reactant_complex = deepcopy(self.reactant)
        reactant_complex.__class__ = Reactant
        product_complex = deepcopy(self.product)
        product_complex.__class__ = Product

        reaction = Reaction(reactant_complex, product_complex)
        reaction.ts = self.ts

        reactions_wc.append(reaction)

        # As with the product complex add the dissociation of the product
        # complex into it's separated components
        if len(self.prods) > 1:
            product_complex = deepcopy(self.product)
            product_complex.__class__ = Reactant
            reactions_wc.append(Reaction(*self.prods, product_complex,
                                         name='product_complex'))

        plot_reaction_profile(reactions=reactions_wc,
                              units=units, name=self.name)

        return None

    def calculate_reaction_profile(self, units=KcalMol, with_complexes=False):
        """
        Calculate and plot a reaction profile for this reaction in some units

        Arguments:
            units (autode.units.Unit):
            with_complexes (bool): Calculate the lowest energy conformers
                                   of the reactant and product complexes
        """
        logger.info('Calculating reaction profile')

        @work_in(self.name)
        def calculate(reaction):
            reaction.find_lowest_energy_conformers()
            reaction.optimise_reacs_prods()
            reaction.find_complexes()
            reaction.locate_transition_state()
            reaction.find_lowest_energy_ts_conformer()
            reaction.calculate_single_points()
            return None

        calculate(self)

        if not with_complexes:
            plot_reaction_profile([self], units=units, name=self.name)

        if with_complexes:
            self._plot_reaction_profile_with_complexes(units=units)

        return None

    def __init__(self, *args, name='reaction', solvent_name=None):
        logger.info(f'Generating a Reaction object for {name}')

        self.name = name
        self.reacs = [mol for mol in args if isinstance(mol, Reactant)]
        self.prods = [mol for mol in args if isinstance(mol, Product)]

        self.reactant, self.product = None, None
        self.ts, self.tss = None, None

        self.type = reaction_types.classify(reactants=self.reacs,
                                            products=self.prods)

        if solvent_name is not None:
            self.solvent = get_solvent(solvent_name=solvent_name)
        else:
            self.solvent = None

        self._check_solvent()
        self._check_balance()
