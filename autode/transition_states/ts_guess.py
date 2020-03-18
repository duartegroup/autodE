from autode.log import logger
from copy import deepcopy
from autode.config import Config
from autode.mol_graphs import get_active_mol_graph
from autode.transition_states.optts import get_displaced_xyzs_along_imaginary_mode
from autode.transition_states.optts import ts_has_correct_imaginary_vector
from autode.species import Species
from autode.calculation import Calculation
from autode.mol_graphs import get_mapping_ts_template
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.templates import template_matches


def get_ts_guess_constrained_opt(reactant, method, keywords, name, distance_consts, product):
    """Get a TS guess from a constrained optimisation with the active atoms fixed at values defined in distance_consts

    Arguments:


    Returns:
       (autode.ts_guess.TSguess):
    """
    logger.info('Getting TS guess from constrained optimisation')

    opt_mol_with_const = deepcopy(reactant)
    const_opt = Calculation(name=f'{name}_constrained_opt', molecule=opt_mol_with_const, method=method,
                            keywords_list=keywords, n_cores=Config.n_cores, distance_constraints=distance_consts)
    const_opt.run()

    # Form a transition state guess from the optimised atoms and set the corrisponding energy
    ts_guess = TSguess(atoms=const_opt.get_final_atoms(), reactant=reactant, product=product)
    ts_guess.energy = const_opt.get_energy()

    return ts_guess


def get_template_ts_guess(reactant, product, bond_rearrangement,  method, keywords, dist_thresh=4.0):
    """Get a transition state guess object by searching though the stored TS templates

    Arguments:
        reactant (mol object): reactant object
        bond_rearrangement (list(tuple)):
        product (mol object): product object
        method (autode.wrappers.base.ElectronicStructureMethod):
        keywords (list(str)): Keywords to use for the ElectronicStructureMethod

    Keyword Arguments:
        dist_thresh (float): distance above which a constrained optimisation probably won't work due to the inital
                             geometry being too far away from the ideal (default: {4.0})

    Returns:
        TSGuess object: ts guess object
    """
    logger.info('Getting TS guess from stored TS template')
    active_bonds_and_dists_ts = {}

    # This will add edges so don't modify in place
    mol_graph = get_active_mol_graph(species=reactant, active_bonds=bond_rearrangement.all)
    ts_guess_templates = get_ts_templates()

    name = f'{reactant.name}_template_{bond_rearrangement}'

    for ts_template in ts_guess_templates:

        if template_matches(mol=reactant, ts_template=ts_template, mol_graph=mol_graph):
            mapping = get_mapping_ts_template(larger_graph=mol_graph, smaller_graph=ts_template.graph)
            for active_bond in bond_rearrangement.all:
                i, j = active_bond
                try:
                    active_bonds_and_dists_ts[active_bond] = ts_template.graph.edges[mapping[i],
                                                                                     mapping[j]]['weight']
                except KeyError:
                    logger.warning(f'Couldn\'t find a mapping for bond {i}-{j}')

            if len(active_bonds_and_dists_ts) == len(bond_rearrangement.all):
                logger.info('Found a TS guess from a template')

                if any([reactant.distance_matrix[bond[0], bond[1]] > dist_thresh for bond in bond_rearrangement.all]):
                    logger.info(f'TS template has => 1 active bond distance larger than {dist_thresh}. Passing')
                    pass
                else:
                    return get_ts_guess_constrained_opt(reactant, method=method, keywords=keywords, name=name,
                                                        distance_consts=active_bonds_and_dists_ts, product=product)

    logger.info('Could not find a TS guess from a template')
    return None


class TSguess(Species):

    def check_optts_convergence(self):

        if not self.optts_calc.optimisation_converged():
            if self.optts_calc.optimisation_nearly_converged():
                logger.info('OptTS nearly did converge. Will try more steps')
                self.optts_nearly_converged = True
                all_xyzs = self.optts_calc.get_final_atoms()
                self.xyzs = all_xyzs[:self.n_atoms]
                if self.qm_solvent_xyzs is not None:
                    self.qm_solvent_xyzs = all_xyzs[self.n_atoms:]
                self.name += '_reopt'
                self.run_optts()
                return

            logger.warning('OptTS calculation was no where near converging')

        else:
            self.optts_converged = True

        return

    def do_displacements(self, size=0.75):
        """Attempts to remove second imaginary mode by displacing either way along it

        Keyword Arguments:
            size (float): magnitude to displace along (default: {0.75})
        """
        mode_lost = False
        imag_freqs = []
        orig_optts_calc = deepcopy(self.optts_calc)
        orig_name = copy(self.name)
        self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, self.n_atoms, displacement_magnitude=size)
        self.name += '_dis'
        self.run_optts()
        if self.calc_failed:
            logger.error('Displacement lost correct imaginary mode, trying backwards displacement')
            mode_lost = True
            self.calc_failed = False

        if not mode_lost:
            self.check_optts_convergence()
            if not self.calc_failed:
                imag_freqs, _, _ = self.get_imag_frequencies_xyzs_energy()
                if len(imag_freqs) > 1:
                    logger.warning(f'OptTS calculation returned {len(imag_freqs)} imaginary frequencies, '
                                   f'trying displacement backwards')
                if len(imag_freqs) == 1:
                    logger.info('Displacement fixed multiple imaginary modes')
                    return
            else:
                logger.error('Displacement lost correct imaginary mode, trying backwards displacement')
                mode_lost = True

        if len(imag_freqs) > 1 or mode_lost:
            self.optts_calc = orig_optts_calc
            self.name = orig_name
            self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, self.n_atoms,
                                                                displacement_magnitude=-size)
            self.name += '_dis2'
            self.run_optts()
            if self.calc_failed:
                logger.error('Displacement lost correct imaginary mode')
                self.calc_failed = True
                return

            imag_freqs, _, _ = self.get_imag_frequencies_xyzs_energy()

            if len(imag_freqs) > 1:
                logger.error('Couldn\'t remove optts_block imaginary frequencies by displacement')

        return

    def run_optts(self, imag_freq_threshold=-50):
        """Runs the optts calc. Calculates a hessian first to the ts guess has the right imaginary mode

        Args:
            imag_freq_threshold (int, optional): If the imaginary mode has a higher (less negative) frequency than this it will not count as the right mode. Defaults to -50.
        """
        logger.info('Getting orca out lines from OptTS calculation')

        if self.qm_solvent_xyzs is not None:
            solvent_atoms = [i for i in range(self.n_atoms, self.n_atoms + len(self.qm_solvent_xyzs))]
        else:
            solvent_atoms = None

        self.hess_calc = Calculation(name=self.name + '_hess', molecule=self, method=self.method,
                                     keywords_list=self.method.keywords.hess, n_cores=Config.n_cores,
                                     partial_hessian=solvent_atoms)

        self.hess_calc.run()

        imag_freqs = self.hess_calc.get_imag_freqs()
        if len(imag_freqs) == 0:
            logger.info('Hessian showed no imaginary modes')
            self.calc_failed = True
            return
        if len(imag_freqs) > 1:
            logger.warning(f'Hessian had {len(imag_freqs)} imaginary modes')
        if imag_freqs[0] > imag_freq_threshold:
            logger.info('Imaginary modes were too small to be significant')
            self.calc_failed = True
            return

        if not ts_has_correct_imaginary_vector(self.hess_calc, n_atoms=self.n_atoms, active_bonds=self.active_bonds,
                                               threshold_contribution=0.1):
            self.calc_failed = True
            return

        self.optts_calc = Calculation(name=self.name + '_optts', molecule=self, method=self.method,
                                      keywords_list=self.method.keywords.opt_ts, n_cores=Config.n_cores,
                                      bond_ids_to_add=self.active_bonds,
                                      partial_hessian=solvent_atoms, other_input_block=self.method.keywords.optts_block,
                                      cartesian_constraints=solvent_atoms)

        self.optts_calc.run()
        all_xyzs = self.optts_calc.get_final_atoms()
        self.xyzs = all_xyzs[:self.n_atoms]
        if self.qm_solvent_xyzs is not None:
            self.qm_solvent_xyzs = all_xyzs[self.n_atoms:]
        return

    def get_imag_frequencies_xyzs_energy(self):
        return self.optts_calc.get_imag_freqs(), self.optts_calc.get_final_atoms(), self.optts_calc.get_energy()

    def get_charges(self):
        return self.optts_calc.get_atomic_charges()

    def __init__(self, atoms, reactant, product, name='ts_guess'):
        """
        Transition state guess

        Arguments:
            atoms (list(autode.atoms.Atom)):
            reactant (autode.complex.ReactantComplex):
            product (autode.complex.ProductComplex):

        Keyword Arguments:
            name (str): name of ts guess (default: {'ts_guess'})
        """
        super().__init__(name=name, atoms=None, charge=reactant.charge, mult=product.mult)
        self.solvent = reactant.solvent
        self.atoms = atoms

        self.reactant = reactant
        self.product = product

        self.optts_calc = None
        self.hess_calc = None

        self.calc_failed = False


class SolvatedTSguess(TSguess):

    def __init__(self, reactant, product, reaction_type, bond_rearrangment, name='ts_guess'):
        super().__init__(reactant, product, reaction_type, bond_rearrangment, name)

        self.qm_solvent_xyzs = None
        self.mm_solvent_xyzs = None

