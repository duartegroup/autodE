from .log import logger
from . import reactions
from . import dissociation
from . import rearrangement
from . import substitution
from .molecule import Reactant
from .molecule import Product
from .units import KcalMol
from .units import KjMol
from .constants import Constants
from .plotting import plot_reaction_profile


class Reaction(object):

    def check_balance(self):

        n_reac_atoms, n_prod_atoms = [reac.n_atoms for reac in self.reacs], [prod.n_atoms for prod in self.prods]
        reac_charges, prod_charges = [r.charge for r in self.reacs], [p.charge for p in self.prods]
        try:
            assert sum(n_reac_atoms) == sum(n_prod_atoms)
            assert sum(reac_charges) == sum(prod_charges)
        except AssertionError:
            logger.critical('Charge and/or number of atoms doesn\'t balance')
            exit()

    def check_solvent(self):
        try:
            assert all([mol.solvent == self.reacs[0].solvent for mol in self.reacs + self.prods])
        except AssertionError:
            logger.critical('Solvents in reactants and products don\'t match')
            exit()

    def switch_addition(self):
        logger.info('Reaction classified as addition. Swapping reacs and prods and switching to dissociation')
        self.type = reactions.Dissociation
        self.prods, self.reacs = self.reacs, self.prods

    def calc_delta_e(self):
        """
        Calculate the ∆Er of a reaction defined as    ∆E = E(products) - E(reactants)
        :return: Energy difference in Hartrees
        """
        logger.info('Calculating ∆Er')
        return sum(filter(None, [p.energy for p in self.prods])) - sum(filter(None, [r.energy for r in self.reacs]))

    def calc_delta_e_ddagger(self):
        """
        Calculate the ∆E‡ of a reaction defined as    ∆E = E(ts) - E(reactants)
        :return: Energy difference in Hartrees
        """
        logger.info('Calculating ∆E‡')
        return self.ts.energy - sum(filter(None, [r.energy for r in self.reacs]))

    def find_lowest_energy_conformers(self):
        """
        Try and locate the lowest energy conformation using RDKit, then optimise them with XTB, then
        optimise the unique (defined by an energy cut-off) conformers with ORCA
        :return:
        """
        for mol in self.reacs + self.prods:
            if mol.n_atoms > 1:
                mol.generate_rdkit_conformers()
                mol.optimise_conformers_xtb()
                mol.strip_non_unique_confs()
                mol.optimise_conformers_orca()
                mol.find_lowest_energy_conformer()

    def optimise_reacs_prods(self):
        logger.info('Calculating optimised reactants and products')
        [mol.optimise() for mol in self.reacs + self.prods]

    def calculate_single_points(self):
        [mol.single_point() for mol in self.reacs + self.prods + [self.ts]]

    def find_lowest_energy_ts(self):
        if len(self.tss) > 1:
            logger.info('Found more than 1 TS. Choosing the lowest energy')
            min_ts_energy = min([ts.energy for ts in self.tss])
            return [ts for ts in self.tss if ts.energy == min_ts_energy][0]
        else:
            return self.tss[0]

    def locate_transition_state(self):

        if self.type == reactions.Dissociation:
            self.ts = dissociation.find_ts(self)

        elif self.type == reactions.Rearrangement:
            self.tss = rearrangement.find_tss(self)
            self.ts = self.find_lowest_energy_ts()

        elif self.type == reactions.Substitution:
            self.ts = substitution.find_ts(self)

        # TODO elimination reactions

        else:
            logger.critical('Reaction type not currently supported')
            exit()

        if self.ts is not None:
            self.ts.charge = sum([reactant.charge for reactant in self.reacs])
            self.ts.mult = sum([reactant.mult for reactant in self.reacs]) - (len(self.reacs) - 1)
            self.ts.solvent = self.reacs[0].solvent

    def calculate_reaction_profile(self, units=KcalMol):
        logger.info('Calculating reaction profile')
        self.find_lowest_energy_conformers()
        self.optimise_reacs_prods()
        self.locate_transition_state()
        self.calculate_single_points()

        conversion = Constants.ha2kJmol if units == KjMol else Constants.ha2kcalmol
        plot_reaction_profile(e_reac=0.0,
                              e_ts=conversion * self.calc_delta_e_ddagger(),
                              e_prod=conversion * self.calc_delta_e(), units=units,
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

        self.check_solvent()
        self.check_balance()

        self.type = reactions.classify(reacs=self.reacs, prods=self.prods)

        if self.type == reactions.Addition:
            self.switch_addition()
