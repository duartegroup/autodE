from autode.log import logger
from autode import reactions
from autode.transition_states.locate_tss import find_tss
from autode.molecule import Molecule
from autode.molecule import SolvatedMolecule
from autode.molecule import Reactant
from autode.molecule import Product
from autode.units import KcalMol
from autode.units import KjMol
from autode.plotting import plot_reaction_profile
from autode.utils import work_in
from autode.solvent.solvents import get_solvent
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.exceptions import UnbalancedReaction
from autode.exceptions import SolventsDontMatch
from copy import deepcopy


def calculate_reaction_profile(reaction, units):
    logger.info('Calculating reaction profile')

    if isinstance(reaction, SolvatedReaction):
        reaction.calc_solvent()

    reaction.find_lowest_energy_conformers()
    reaction.optimise_reacs_prods()
    reaction.locate_transition_state()
    reaction.find_lowest_energy_ts_conformer()
    reaction.calculate_single_points()

    plot_reaction_profile(e_reac=0.0,
                          e_ts=reaction.calc_delta_e_ddagger(units=units),
                          e_prod=reaction.calc_delta_e(units=units),
                          units=units,
                          reacs=reaction.reacs,
                          prods=reaction.prods,
                          ts=reaction.ts,
                          switched=reaction.switched_reacs_prods)
    return None


class Reaction:

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
        self.switched_reacs_prods = True

    def calc_delta_e(self, units=KcalMol):
        """Calculate the ∆Er of a reaction defined as    ∆E = E(products) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')
        products_energy = sum(filter(None, [p.energy for p in self.prods]))
        reactants_energy = sum(filter(None, [r.energy for r in self.reacs]))

        return units.conversion * (products_energy - reactants_energy)

    def calc_delta_e_ddagger(self, units=KcalMol):
        """Calculate the ∆E‡ of a reaction defined as    ∆E = E(ts) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        if self.ts is None:
            return None

        if self.ts.energy is None:
            logger.error('TS had no energy. Setting ∆E‡ = None')
            return None

        return units.conversion * (self.ts.energy - sum(filter(None, [r.energy for r in self.reacs])))

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

    @work_in('conformers')
    def find_lowest_energy_conformers(self):
        """Try and locate the lowest energy conformation using simulated annealing, then optimise them with xtb, then
        optimise the unique (defined by an energy cut-off) conformers with an electronic structure method"""

        for mol in self.reacs + self.prods:
            mol.find_lowest_energy_conformer(low_level_method=get_lmethod(), high_level_method=get_hmethod())

    @work_in('reactants_and_products')
    def optimise_reacs_prods(self):
        """Perform a geometry optimisation on all the reactants and products using the hcode"""
        h_method = get_hmethod()
        logger.info(f'Calculating optimised reactants and products with {h_method.name}')
        [mol.optimise(method=h_method) for mol in self.reacs + self.prods]

    @work_in('transition_states')
    def locate_transition_state(self):

        self.tss = find_tss(self)
        self.ts = self.find_lowest_energy_ts()

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
        """Perform a single point energy evaluations on all the reactants and products using the hcode"""
        molecules = self.reacs + self.prods + [self.ts]
        [mol.single_point(method=get_hmethod()) for mol in molecules if mol is not None]

    def calculate_reaction_profile(self, units=KcalMol):
        return calculate_reaction_profile(self, units=units)

    def __init__(self, mol1=None, mol2=None, mol3=None, mol4=None, mol5=None, mol6=None, name='reaction',
                 solvent_name=None):
        logger.info(f'Generating a Reaction object for {name}')

        self.name = name
        molecules = [mol1, mol2, mol3, mol4, mol5, mol6]
        self.reacs = [mol for mol in molecules if isinstance(mol, Reactant) and mol is not None]
        self.prods = [mol for mol in molecules if isinstance(mol, Product) and mol is not None]

        self.ts, self.tss = None, None

        self.type = reactions.classify(reactants=self.reacs, products=self.prods)

        self.solvent = get_solvent(solvent_name=solvent_name) if solvent_name is not None else None

        self._check_solvent()
        self._check_balance()

        self.switched_reacs_prods = False               #: Have the reactants and products been switched
        # If there are more bonds in the product e.g. an addition reaction then switch as the TS is then easier to find
        if sum(p.graph.number_of_edges() for p in self.prods) > sum(r.graph.number_of_edges() for r in self.reacs):
            self.switch_reactants_products()

        if self.type == reactions.Rearrangement:
            self._check_rearrangement()


class SolvatedReaction(Reaction):

    def calc_delta_e_ddagger(self, units=KcalMol):
        """Calculate the ∆E‡ of a reaction defined as    ∆E = E(ts) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        if self.ts is None:
            return None

        if self.ts.energy is None:
            logger.error('TS had no energy. Setting ∆E‡ = None')
            return None

        return units.conversion * (self.ts.energy - sum(filter(None, [r.energy for r in self.reacs])) + ((len(self.reacs) - 1) * self.solvent_mol.energy))

    def calc_delta_e(self, units=KcalMol):
        """Calculate the ∆Er of a reaction defined as    ∆E = E(products) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')
        products_energy = sum(filter(None, [p.energy for p in self.prods]))
        reactants_energy = sum(filter(None, [r.energy for r in self.reacs]))

        return units.conversion * (products_energy - reactants_energy + ((len(self.reacs) - len(self.prods)) * self.solvent_mol.energy))

    @work_in('solvent')
    def calc_solvent(self):
        logger.info('Optimising the solvent molecule')
        self.solvent_mol = SolvatedMolecule(name=self.solvent.name, smiles=self.solvent.smiles)
        self.solvent_mol.find_lowest_energy_conformer(low_level_method=get_lmethod())
        self.solvent_mol.optimise(get_hmethod())
        for mol in self.reacs + self.prods:
            mol.solvent_mol = self.solvent_mol

    def make_solvated_mol_objects(self):
        solvated_reacs, solvated_prods = [], []
        for mol in self.reacs:
            solvated_mol = SolvatedMolecule(name=mol.name, atoms=mol.atoms, charge=mol.charge, mult=mol.mult)
            solvated_mol.smiles = mol.smiles
            solvated_mol.rdkit_mol_obj = mol.rdkit_mol_obj
            solvated_mol.rdkit_conf_gen_is_fine = mol.rdkit_conf_gen_is_fine
            solvated_mol.graph = deepcopy(mol.graph)
            solvated_reacs.append(solvated_mol)
        self.reacs = solvated_reacs
        for mol in self.prods:
            solvated_mol = SolvatedMolecule(name=mol.name, atoms=mol.atoms, charge=mol.charge, mult=mol.mult)
            solvated_mol.smiles = mol.smiles
            solvated_mol.rdkit_mol_obj = mol.rdkit_mol_obj
            solvated_mol.rdkit_conf_gen_is_fine = mol.rdkit_conf_gen_is_fine
            solvated_mol.graph = deepcopy(mol.graph)
            solvated_prods.append(solvated_mol)
        self.prods = solvated_prods

    def __init__(self, mol1=None, mol2=None, mol3=None, mol4=None, mol5=None, mol6=None, name='reaction',
                 solvent_name=None):
        super().__init__(mol1, mol2, mol3, mol4, mol5, mol6, name, solvent_name)

        self.solvent_mol = None

        self.make_solvated_mol_objects()
