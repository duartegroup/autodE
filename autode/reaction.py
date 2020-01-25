from autode.log import logger
from autode import reactions
from autode.transition_states.locate_tss import find_tss
from autode.molecule import Reactant
from autode.molecule import Product
from autode.units import KcalMol
from autode.units import KjMol
from autode.constants import Constants
from autode.plotting import plot_reaction_profile
from autode.mol_graphs import get_mapping
from autode.utils import work_in
from autode.config import Config
from copy import deepcopy
import os


class Reaction:

    def check_balance(self):
        """Check that the number of atoms and charge balances between reactants and products. If they don't exit
        immediately
        """

        n_reac_atoms, n_prod_atoms = [reac.n_atoms for reac in self.reacs], [prod.n_atoms for prod in self.prods]
        reac_charges, prod_charges = [r.charge for r in self.reacs], [p.charge for p in self.prods]
        if sum(n_reac_atoms) != sum(n_prod_atoms):
            logger.critical('Number of atoms doesn\'t balance')
            exit()
        if sum(reac_charges) != sum(prod_charges):
            logger.critical('Charge doesn\'t balance')
            exit()
        self.charge = sum(reac_charges)

    def check_solvent(self):
        """Check that all the solvents are the same for reactants and products
        """
        if not all([mol.solvent == self.reacs[0].solvent for mol in self.reacs + self.prods]):
            logger.critical('Solvents in reactants and products don\'t match')
            exit()

    def set_solvent(self, solvent):
        if solvent is not None:
            logger.info(f'Setting solvent as {solvent}')

            assert type(solvent) == str
            for mol in self.reacs + self.prods:
                mol.solvent = solvent

    def switch_addition(self):
        """Addition reactions are hard to find the TSs for, so swap reactants and products and classify as dissociation
        """
        logger.info('Reaction classified as addition. Swapping reacs and prods and switching to dissociation')
        self.type = reactions.Dissociation
        self.prods, self.reacs = self.reacs, self.prods
        self.switched_reacs_prods = True

    def check_rearrangement(self):
        """Could be an intramolecular addition, so will swap reactants and products if this is the case"""

        logger.info('Reaction classified as rearrangement, checking if it is an intramolecular addition')
        reac_bonds_list = [mol.n_bonds for mol in self.reacs]
        prod_bonds_list = [mol.n_bonds for mol in self.prods]
        delta_n_bonds = sum(reac_bonds_list) - sum(prod_bonds_list)

        if delta_n_bonds < 0:
            logger.info('Products have more bonds than the reactants, swapping reacs and prods and going in reverse')
            self.prods, self.reacs = self.reacs, self.prods
        else:
            logger.info('Does not appear to be an intramolecular addition, continuing')

    def calc_delta_e(self):
        """Calculate the ∆Er of a reaction defined as    ∆E = E(products) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')
        return sum(filter(None, [p.energy for p in self.prods])) - sum(filter(None, [r.energy for r in self.reacs]))

    def calc_delta_e_ddagger(self):
        """Calculate the ∆E‡ of a reaction defined as    ∆E = E(ts) - E(reactants)

        Returns:
            float: energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        if self.ts.energy is not None:
            return self.ts.energy - sum(filter(None, [r.energy for r in self.reacs]))
        else:
            logger.error('TS had no energy. Setting ∆E‡ = None')
            return None

    @work_in('conformers')
    def find_lowest_energy_conformers(self):
        """Try and locate the lowest energy conformation using RDKit, then optimise them with XTB, then
        optimise the unique (defined by an energy cut-off) conformers with an electronic structure method"""

        for mol in self.reacs + self.prods:
            if mol.n_atoms > 1:
                mol.find_lowest_energy_conformer()

    @work_in('optimise_reactants_and_products')
    def optimise_reacs_prods(self):
        """Perform a geometry optimisation on all the reactants and products using the hcode
        """
        logger.info('Calculating optimised reactants and products')
        [mol.optimise() for mol in self.reacs + self.prods]

    def find_lowest_energy_ts(self):
        """From all the transition state objects in Reaction.tss choose the lowest energy if there is more than one
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

    @work_in('tss')
    def locate_transition_state(self):

        # Clear the PES graphs, so they don't write over each other
        file_extension = Config.image_file_extension
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(file_extension):
                os.remove(filename)

        self.tss = find_tss(self)
        self.ts = self.find_lowest_energy_ts()

    @work_in('ts_confs')
    def ts_confs(self):
        """Find the lowest energy conformer of the transition state"""
        logger.info('Finding all the stereocentres in the transition state')

        stereocentres = set()
        n_atoms = 0
        for reac in self.reacs:
            if reac.stereocentres is not None:
                for stereocentre in reac.stereocentres:
                    stereocentres.add(stereocentre + n_atoms)
            n_atoms += reac.n_atoms

        for mol in self.prods:
            if mol.stereocentres is not None:
                mapping = get_mapping(self.product_graph, mol.graph)[0]
                for ts_index, mol_index in mapping.items():
                    if mol_index in mol.stereocentres:
                        stereocentres.add(ts_index)

        self.ts.stereocentres = sorted(stereocentres)

        ts_copy = deepcopy(self.ts)
        logger.info('Trying to find lowest energy TS conformer')

        self.ts = self.ts.find_lowest_energy_conformer()
        if self.ts is None:
            logger.error('Conformer search lost the TS, using the original TS')
            self.ts = ts_copy
        elif self.ts.energy > ts_copy.energy:
            logger.error(f'Conformer search increased the TS energy by {(self.ts.energy - ts_copy.energy):.3g} Hartree')
            self.ts = ts_copy

    @work_in('single_points')
    def calculate_single_points(self):
        """Perform a single point energy evaluations on all the reactants and products using the hcode"""
        molecules = self.reacs + self.prods + [self.ts]
        [mol.single_point() for mol in molecules if mol is not None]

    def calculate_reaction_profile(self, units=KcalMol):
        logger.info('Calculating reaction profile')

        @work_in(self.name)
        def calc_reaction_profile():
            self.find_lowest_energy_conformers()
            self.optimise_reacs_prods()
            self.locate_transition_state()
            if self.ts is not None:
                self.ts_confs()
            self.calculate_single_points()

            if self.ts is None:
                return logger.error('TS is None – cannot plot a reaction profile')

            conversion = Constants.ha2kJmol if units == KjMol else Constants.ha2kcalmol
            plot_reaction_profile(e_reac=0.0,
                                  e_ts=conversion * self.calc_delta_e_ddagger(),
                                  e_prod=conversion * self.calc_delta_e(),
                                  units=units,
                                  reacs=self.reacs,
                                  prods=self.prods,
                                  is_true_ts=self.ts.is_true_ts(),
                                  ts_is_converged=self.ts.converged,
                                  switched=self.switched_reacs_prods)

        calc_reaction_profile()

        return None

    def __init__(self, mol1=None, mol2=None, mol3=None, mol4=None, mol5=None, mol6=None, name='reaction', solvent=None):
        logger.info(f'Generating a Reaction object for {name}')

        self.name = name
        molecules = [mol1, mol2, mol3, mol4, mol5, mol6]
        self.reacs = [mol for mol in molecules if isinstance(mol, Reactant) and mol is not None]
        self.prods = [mol for mol in molecules if isinstance(mol, Product) and mol is not None]
        self.ts, self.tss = None, []
        self.charge = None

        self.type = reactions.classify(reacs=self.reacs, prods=self.prods)

        self.set_solvent(solvent)
        self.check_solvent()
        self.check_balance()

        self.product_graph = None                       #: Full graph of the products

        self.switched_reacs_prods = False               #: Have the reactants and products been switched to
        if self.type == reactions.Addition:
            self.switch_addition()

        if self.type == reactions.Rearrangement:
            self.check_rearrangement()
