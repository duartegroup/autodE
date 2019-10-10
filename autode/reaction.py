from autode.log import logger
from autode import reactions
from autode.transition_states.locate_tss import find_tss
from autode.molecule import Reactant
from autode.molecule import Product
from autode.units import KcalMol
from autode.units import KjMol
from autode.constants import Constants
from autode.plotting import plot_reaction_profile
import os


class Reaction:

    def check_balance(self):
        """
        Check that the number of atoms and charge balances between reactants and products. If they don't exit
        immediately
        :return: None
        """

        n_reac_atoms, n_prod_atoms = [reac.n_atoms for reac in self.reacs], [prod.n_atoms for prod in self.prods]
        reac_charges, prod_charges = [r.charge for r in self.reacs], [p.charge for p in self.prods]
        if sum(n_reac_atoms) != sum(n_prod_atoms):
            logger.critical('Number of atoms doesn\'t balance')
            exit()
        if sum(reac_charges) != sum(prod_charges):
            logger.critical('Charge doesn\'t balance')
            exit()

    def check_solvent(self):
        """
        Check that all the solvents are the same for reactants and products
        :return: None
        """
        if not all([mol.solvent == self.reacs[0].solvent for mol in self.reacs + self.prods]):
            logger.critical('Solvents in reactants and products don\'t match')
            exit()

    def switch_addition(self):
        """
        Addition reactions are hard to find the TSs for so swap reactants and products and classify as dissociation
        :return: None
        """
        logger.info('Reaction classified as addition. Swapping reacs and prods and switching to dissociation')
        self.type = reactions.Dissociation
        self.prods, self.reacs = self.reacs, self.prods

    def calc_delta_e(self):
        """
        Calculate the ∆Er of a reaction defined as    ∆E = E(products) - E(reactants)
        :return: (float) energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')
        return sum(filter(None, [p.energy for p in self.prods])) - sum(filter(None, [r.energy for r in self.reacs]))

    def calc_delta_e_ddagger(self):
        """
        Calculate the ∆E‡ of a reaction defined as    ∆E = E(ts) - E(reactants)
        :return: (float) Energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        if self.ts.energy is not None:
            return self.ts.energy - sum(filter(None, [r.energy for r in self.reacs]))
        else:
            logger.error('TS had no energy. Setting ∆E‡ = None')
            return None

    def find_lowest_energy_conformers(self):
        """
        Try and locate the lowest energy conformation using RDKit, then optimise them with XTB, then
        optimise the unique (defined by an energy cut-off) conformers with an electronic structure method
        :return: None
        """
        here = os.getcwd()
        conformers_directory_path = os.path.join(here, self.name + '_conformers')
        if not os.path.isdir(conformers_directory_path):
            os.mkdir(conformers_directory_path)
            logger.info(f'Creating directory to store conformer output files at {conformers_directory_path:}')
        os.chdir(conformers_directory_path)
        
        for mol in self.reacs + self.prods:
            if mol.n_atoms > 1:
                mol.find_lowest_energy_conformer()

        os.chdir(here)

    def optimise_reacs_prods(self):
        """
        Perform a geometry optimisation on all the reactants and products using the hcode
        :return: None
        """

        logger.info('Calculating optimised reactants and products')
        [mol.optimise() for mol in self.reacs + self.prods]

    def calculate_single_points(self):
        """
        Perform a single point energy evaluations on all the reactants and products using the hcode
        :return: None
        """
        here = os.getcwd()
        single_points_directory_path = os.path.join(here, self.name + '_single_points')
        if not os.path.isdir(single_points_directory_path):
            os.mkdir(single_points_directory_path)
            logger.info(f'Creating directory to store conformer output files at {single_points_directory_path:}')
        os.chdir(single_points_directory_path)

        molecules = self.reacs + self.prods + [self.ts]
        [mol.single_point() for mol in molecules if mol is not None]

    def find_lowest_energy_ts(self):
        """
        From all the transition state objects in Reaction.tss choose the lowest energy if there is more than one
        otherwise return the single transtion state or None if there no TS objects.
        :return:
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

    def locate_transition_state(self):
        here = os.getcwd()
        tss_directory_path = os.path.join(here, self.name + '_tss')
        if not os.path.isdir(tss_directory_path):
            os.mkdir(tss_directory_path)
            logger.info(f'Creating directory to store conformer output files at {tss_directory_path:}')
        os.chdir(tss_directory_path)

        self.tss = find_tss(self)
        self.ts = self.find_lowest_energy_ts()

        os.chdir(here)

    def calculate_reaction_profile(self, units=KcalMol):
        logger.info('Calculating reaction profile')
        self.find_lowest_energy_conformers()
        self.optimise_reacs_prods()
        self.locate_transition_state()
        self.calculate_single_points()

        if self.ts is None:
            return logger.error('TS is None – cannot plot a reaction profile')

        conversion = Constants.ha2kJmol if units == KjMol else Constants.ha2kcalmol
        plot_reaction_profile(e_reac=0.0,
                              e_ts=conversion * self.calc_delta_e_ddagger(),
                              e_prod=conversion * self.calc_delta_e(),
                              units=units,
                              name=(' + '.join([r.name for r in self.reacs]) + ' → ' +
                                    ' + '.join([p.name for p in self.prods])),
                              is_true_ts=self.ts.is_true_ts(),
                              ts_is_converged=self.ts.converged
                              )

    def __init__(self, mol1=None, mol2=None, mol3=None, mol4=None, mol5=None, mol6=None, name='reaction'):
        logger.info('Generating a Reaction object for {}'.format(name))

        self.name = name
        molecules = [mol1, mol2, mol3, mol4, mol5, mol6]
        self.reacs = [mol for mol in molecules if isinstance(mol, Reactant) and mol is not None]
        self.prods = [mol for mol in molecules if isinstance(mol, Product) and mol is not None]
        self.ts, self.tss = None, []

        self.type = reactions.classify(reacs=self.reacs, prods=self.prods)

        self.check_solvent()
        self.check_balance()

        if self.type == reactions.Addition:
            self.switch_addition()
