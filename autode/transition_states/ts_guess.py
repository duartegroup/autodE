from autode.log import logger
from autode.config import Config
from autode.transition_states.optts import get_displaced_xyzs_along_imaginary_mode
from autode.transition_states.optts import ts_has_correct_imaginary_vector
from autode.calculation import Calculation


class TSguess:

    def check_optts_convergence(self):

        if not self.optts_calc.optimisation_converged():
            if self.optts_calc.optimisation_nearly_converged():
                logger.info('OptTS nearly did converge. Will try more steps')
                self.optts_nearly_converged = True
                self.xyzs = self.optts_calc.get_final_xyzs()
                self.name += '_reopt'
                return self.run_orca_optts()

            logger.warning('OptTS calculation was no where near converging')

        else:
            self.optts_converged = True

        return None

    def do_displacements(self):
        self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, displacement_magnitude=1.0)
        self.name += '_dis'
        self.run_orca_optts()

        imag_freqs, ts_xyzs, ts_energy = self.get_imag_frequencies_xyzs_energy()
        self.check_optts_convergence()

        if len(imag_freqs) > 1:
            logger.warning('OptTS calculation returned {} imaginary frequencies'.format(len(imag_freqs)))
            self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, displacement_magnitude=-1.0)
            self.name += '_dis2'
            self.run_orca_optts()
            self.check_optts_convergence()

            if len(imag_freqs) > 1:
                logger.error('Couldn\'t remove other imaginary frequencies by displacement')

        return None

    def run_orca_optts(self):
        logger.info('Getting ORCA out lines from OptTS calculation')

        self.hess_calc = Calculation(name=self.name +'_hess', molecule=self, method=self.method,
                                      keywords=self.method.hess_keywords, n_cores=Config.n_cores,
                                      max_core_mb=Config.max_core)
        
        self.hess_calc.run()
        if not ts_has_correct_imaginary_vector(self.hess_calc, n_atoms=self.n_atoms,
                                           active_bonds=self.active_bonds, threshold_contribution=0.1):
            return None

        self.optts_calc = Calculation(name=self.name + '_optts', molecule=self, method=self.method,
                                      keywords=self.method.opt_ts_keywords, n_cores=Config.n_cores,
                                      max_core_mb=Config.max_core, bond_ids_to_add=self.active_bonds,
                                      optts_block=self.method.opt_ts_block)

        self.optts_calc.run()
        self.xyzs = self.optts_calc.get_final_xyzs()
        return None

    def get_imag_frequencies_xyzs_energy(self):
        return self.optts_calc.get_imag_freqs(), self.optts_calc.get_final_xyzs(), self.optts_calc.get_energy()

    def __init__(self, name='ts_guess', molecule=None, reaction_class=None, active_bonds=None):
        """
        :param name: (str)
        :param reaction_class: (object)
        :param molecule: (object)

        :param active_bonds:
        """
        self.name = name
        self.xyzs = molecule.xyzs
        self.n_atoms = len(molecule.xyzs) if molecule.xyzs is not None else None
        self.reaction_class = reaction_class
        self.solvent = molecule.solvent
        self.charge = molecule.charge
        self.mult = molecule.mult
        self.active_bonds = active_bonds
        self.method = molecule.method

        self.optts_converged = False
        self.optts_nearly_converged = False
        self.optts_calc = None
        self.hess_calc = None
