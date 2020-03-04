from autode.log import logger
from autode.config import Config
from autode.transition_states.optts import get_displaced_xyzs_along_imaginary_mode
from autode.transition_states.optts import ts_has_correct_imaginary_vector
from autode.calculation import Calculation
from autode.geom import xyz2coord
from autode.mol_graphs import make_graph
from copy import deepcopy
from copy import copy


class TSguess:

    def get_bonded_atoms_to_i(self, atom_i):
        bonded_atoms = []
        for edge in self.graph.edges():
            if edge[0] == atom_i:
                bonded_atoms.append(edge[1])
            if edge[1] == atom_i:
                bonded_atoms.append(edge[0])
        return bonded_atoms

    def check_optts_convergence(self):

        if not self.optts_calc.optimisation_converged():
            if self.optts_calc.optimisation_nearly_converged():
                logger.info('OptTS nearly did converge. Will try more steps')
                self.optts_nearly_converged = True
                all_xyzs = self.optts_calc.get_final_xyzs()
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

    def do_displacements(self, magnitude=0.75):
        """Attempts to remove second imaginary mode by displacing either way along it

        Keyword Arguments:
            magnitude (float): magnitude to displace along (default: {0.75})
        """
        mode_lost = False
        imag_freqs = []
        orig_optts_calc = deepcopy(self.optts_calc)
        orig_name = copy(self.name)
        self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, self.n_atoms, displacement_magnitude=magnitude)
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
                    logger.warning(f'OptTS calculation returned {len(imag_freqs)} imaginary frequencies, trying displacement backwards')
                if len(imag_freqs) == 1:
                    logger.info('Displacement fixed multiple imaginary modes')
                    return
            else:
                logger.error('Displacement lost correct imaginary mode, trying backwards displacement')
                mode_lost = True

        if len(imag_freqs) > 1 or mode_lost:
            self.optts_calc = orig_optts_calc
            self.name = orig_name
            self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, self.n_atoms, displacement_magnitude=-1 * magnitude)
            self.name += '_dis2'
            self.run_optts()
            if self.calc_failed:
                logger.error('Displacement lost correct imaginary mode')
                self.calc_failed = True
                return

            imag_freqs, _, _ = self.get_imag_frequencies_xyzs_energy()

            if len(imag_freqs) > 1:
                logger.error('Couldn\'t remove other imaginary frequencies by displacement')

        return

    def run_optts(self, imag_freq_threshold=-50):
        """Runs the optts calc
        """
        logger.info('Getting ORCA out lines from OptTS calculation')

        if self.qm_solvent_xyzs is not None:
            solvent_atoms = [i for i in range(self.n_atoms, self.n_atoms + len(self.qm_solvent_xyzs))]
        else:
            solvent_atoms = None

        self.hess_calc = Calculation(name=self.name + '_hess', molecule=self, method=self.method,
                                     keywords=self.method.hess_keywords, n_cores=Config.n_cores,
                                     max_core_mb=Config.max_core, charges=self.point_charges,
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

        if not ts_has_correct_imaginary_vector(self.hess_calc, n_atoms=self.n_atoms, active_bonds=self.active_bonds, threshold_contribution=0.1):
            self.calc_failed = True
            return

        self.optts_calc = Calculation(name=self.name + '_optts', molecule=self, method=self.method,
                                      keywords=self.method.opt_ts_keywords, n_cores=Config.n_cores,
                                      max_core_mb=Config.max_core, bond_ids_to_add=self.active_bonds,
                                      partial_hessian=solvent_atoms, charges=self.point_charges,
                                      optts_block=self.method.opt_ts_block, cartesian_constraints=solvent_atoms)

        self.optts_calc.run()
        all_xyzs = self.optts_calc.get_final_xyzs()
        self.xyzs = all_xyzs[:self.n_atoms]
        if self.qm_solvent_xyzs is not None:
            self.qm_solvent_xyzs = all_xyzs[self.n_atoms:]
        return

    def get_imag_frequencies_xyzs_energy(self):
        return self.optts_calc.get_imag_freqs(), self.optts_calc.get_final_xyzs(), self.optts_calc.get_energy()

    def get_coords(self):
        return xyz2coord(self.xyzs)

    def get_charges(self):
        return self.optts_calc.get_atomic_charges()

    def __init__(self, name='ts_guess', molecule=None, reaction_class=None, active_bonds=None, reactant=None, product=None):
        """
        Keyword Arguments:
            name (str): name of ts guess (default: {'ts_guess'})
            molecule (molecule object): molecule to base ts guess off (default: {None})
            reaction_class (object): reaction type (reactions.py) (default: {None})
            active_bonds (list(tuples)): list of bonds being made/broken (default: {None})
            reactant (molecule object): reactant object (default: {None})
            product (molecule object): product object (default: {None})
        """
        self.name = name

        if molecule is None:
            logger.error('A TSguess needs a molecule object to initialise')
            return

        self.xyzs = molecule.xyzs
        self.n_atoms = len(molecule.xyzs) if molecule.xyzs is not None else None
        self.reaction_class = reaction_class
        self.solvent = molecule.solvent
        self.charge = molecule.charge
        self.mult = molecule.mult
        self.active_bonds = active_bonds
        self.method = molecule.method
        self.reactant = reactant
        self.product = product
        self.graph = make_graph(self.xyzs, self.n_atoms)
        self.charges = molecule.charges
        self.stereocentres = molecule.stereocentres
        self.qm_solvent_xyzs = molecule.qm_solvent_xyzs
        self.mm_solvent_xyzs = molecule.mm_solvent_xyzs

        self.optts_converged = False
        self.optts_nearly_converged = False
        self.optts_calc = None
        self.hess_calc = None

        self.calc_failed = False

        self.point_charges = None
