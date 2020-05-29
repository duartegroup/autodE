import base64
import hashlib
from autode.solvent.solvents import get_solvent
from autode.transition_states.locate_tss import find_tss
from autode.exceptions import UnbalancedReaction
from autode.exceptions import SolventsDontMatch
from autode.log import logger
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.molecule import Product
from autode.molecule import Reactant
from autode.plotting import plot_reaction_profile
from autode.units import KcalMol
from autode.utils import work_in
from autode import reactions


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
        """Check that the number of atoms and charge balances between reactants and products. If they don't exit
        immediately
        """

        n_reac_atoms, n_prod_atoms = [reac.n_atoms for reac in self.reacs], [prod.n_atoms for prod in self.prods]
        reac_charges, prod_charges = [r.charge for r in self.reacs], [p.charge for p in self.prods]
        if sum(n_reac_atoms) != sum(n_prod_atoms):
            logger.critical('Number of atoms doesn\'t balance')
            raise UnbalancedReaction

        if sum(reac_charges) != sum(prod_charges):
            logger.critical('Charge doesn\'t balance')
            raise UnbalancedReaction

        self.charge = sum(reac_charges)

    def _check_solvent(self):
        """Check that all the solvents are the same for reactants and products
        """
        if self.solvent is None:
            if all([mol.solvent is None for mol in self.reacs + self.prods]):
                logger.info('Reaction is in the gas phase')
                return

            elif all([mol.solvent is not None for mol in self.reacs + self.prods]):
                if not all([mol.solvent == self.reacs[0].solvent for mol in self.reacs + self.prods]):
                    logger.critical('Solvents in reactants and products don\'t match')
                    raise SolventsDontMatch

                else:
                    logger.info(f'Setting the reaction solvent to {self.reacs[0].solvent}')
                    self.solvent = self.reacs[0].solvent

            else:
                print([mol.solvent for mol in self.reacs + self.prods])
                print([mol.name for mol in self.reacs + self.prods])

                raise SolventsDontMatch

        if self.solvent is not None:
            logger.info(f'Setting solvent to {self.solvent.name} for all molecules in the reaction')

            for mol in self.reacs + self.prods:
                mol.solvent = self.solvent

        logger.info(f'Set the solvent of all species in the reaction to {self.solvent.name}')
        return None

    def _check_rearrangement(self):
        """Could be an intramolecular addition, so will swap reactants and products if this is the case"""

        logger.info('Reaction classified as rearrangement, checking if it is an intramolecular addition')
        reac_bonds_list = [mol.graph.number_of_edges() for mol in self.reacs]
        prod_bonds_list = [mol.graph.number_of_edges() for mol in self.prods]
        delta_n_bonds = sum(reac_bonds_list) - sum(prod_bonds_list)

        if delta_n_bonds < 0:
            logger.info('Products have more bonds than the reactants, swapping reacs and prods and going in reverse')
            self.prods, self.reacs = self.reacs, self.prods
            self.switched_reacs_prods = True
        else:
            logger.info('Does not appear to be an intramolecular addition, continuing')

    def switch_reactants_products(self):
        """Addition reactions are hard to find the TSs for, so swap reactants and products and classify as dissociation
        """
        logger.info('Reaction classified as addition. Swapping reacs and prods and switching to dissociation')
        if self.type == reactions.Addition:
            self.type = reactions.Dissociation

        self.prods, self.reacs = self.reacs, self.prods

    def calc_delta_e(self):
        """Calculate the ∆Er of a reaction defined as    ∆E = E(products) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')

        if any(mol.energy is None for mol in self.reacs + self.prods):
            logger.error('Cannot calculate ∆Er. At least one required energy was None')
            return None

        return sum([p.energy for p in self.prods]) - sum([r.energy for r in self.reacs])

    def calc_delta_e_ddagger(self):
        """Calculate the ∆E‡ of a reaction defined as    ∆E = E(ts) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        if self.ts is None:
            logger.error('No TS, cannot calculate ∆E‡')
            return None

        if self.ts.energy is None or any(r.energy is None for r in self.reacs):
            logger.error('TS or a reactant had no energy, cannot calculate ∆E‡')
            return None

        return self.ts.energy - sum([r.energy for r in self.reacs])

    def find_lowest_energy_ts(self):
        """From all the transition state objects in Reaction.pes1d choose the lowest energy if there is more than one
        otherwise return the single transtion state or None if there no TS objects.
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

    def find_lowest_energy_conformers(self, calc_reacs=True, calc_prods=True):
        """Try and locate the lowest energy conformation using simulated
        annealing, then optimise them with xtb, then optimise the unique
        (defined by an energy cut-off) conformers with an electronic structure
        method"""

        molecules = []
        if calc_reacs:
            molecules += self.reacs
        if calc_prods:
            molecules += self.prods

        for mol in molecules:
            mol.find_lowest_energy_conformer(lmethod=get_lmethod(),
                                             hmethod=get_hmethod())

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

    @work_in('transition_states')
    def locate_transition_state(self):

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

    def calculate_reaction_profile(self, units=KcalMol):
        logger.info('Calculating reaction profile')

        @work_in(self.name)
        def calculate(reaction):

            reaction.find_lowest_energy_conformers()
            reaction.optimise_reacs_prods()
            reaction.locate_transition_state()
            reaction.find_lowest_energy_ts_conformer()
            reaction.calculate_single_points()

            plot_reaction_profile(reactions=[reaction], units=units, name=self.name)
            return None

        return calculate(self)

    def __init__(self, *args, name='reaction', solvent_name=None):
        logger.info(f'Generating a Reaction object for {name}')

        self.name = name
        self.reacs = [mol for mol in args if isinstance(mol, Reactant)]
        self.prods = [mol for mol in args if isinstance(mol, Product)]

        self.ts, self.tss = None, None

        self.type = reactions.classify(reactants=self.reacs, products=self.prods)

        if solvent_name is not None:
            self.solvent = get_solvent(solvent_name=solvent_name)
        else:
            self.solvent = None

        self._check_solvent()
        self._check_balance()

        if self.type == reactions.Rearrangement:
            self._check_rearrangement()


class MultiStepReaction:

    def calculate_reaction_profile(self, units=KcalMol):
        """Calculate a multistep reaction profile using the products of step 1
        as the reactants of step 2 etc."""
        logger.info('Calculating reaction profile')

        @work_in(self.name)
        def calculate(reaction, calc_reac_conformers=False):

            # If the step is > 1 then there is no need to calculate the
            # conformers of the reactants..
            reaction.find_lowest_energy_conformers(calc_prods=True,
                                                   calc_reacs=calc_reac_conformers)

            reaction.optimise_reacs_prods()
            reaction.locate_transition_state()
            reaction.find_lowest_energy_ts_conformer()
            reaction.calculate_single_points()

            return None

        def check_reaction(current_reaction, previous_reaction):
            """Check that the reactants of the current reaction are the same
            as the previous products. NOT exhaustive"""

            assert len(current_reaction.reacs) == len(previous_reaction.prods)
            n_reacting_atoms = sum(reac.n_atoms for reac in current_reaction.reacs)
            n_prev_product_atoms = sum(prod.n_atoms for prod in previous_reaction.prods)
            assert n_reacting_atoms == n_prev_product_atoms

        for i, r in enumerate(self.reactions):
            r.name = f'{self.name}_step{i}'

            if i == 0:
                # First reaction requires calculating reactant conformers
                calculate(reaction=r, calc_reac_conformers=True)

            else:
                # Set the reactants of this reaction as the previous set of
                # products and don't recalculate conformers
                check_reaction(current_reaction=r,
                               previous_reaction=self.reactions[i-1])
                r.reacs = self.reactions[i-1].prods
                calculate(reaction=r)

        plot_reaction_profile(self.reactions, units=units, name=self.name)
        return None

    def __init__(self, *args, name='reaction'):
        """
        Reaction with multiple steps

        Arguments:
            *args (autode.reaction.Reaction): Set of reactions to calculate the
                                              reaction profile for
        """
        self.name = str(name)
        self.reactions = args

        assert all(type(reaction) is Reaction for reaction in self.reactions)
