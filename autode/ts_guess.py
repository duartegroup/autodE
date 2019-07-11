from .log import logger
from .config import Config
from .optts import get_displaced_xyzs_along_imaginary_mode
from .optts import optts_converged
from .optts import is_optts_nearly_converged
from .ORCAio import get_imag_frequencies_xyzs_energy_optts
from .ORCAio import get_orca_opt_final_xyzs
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca


class TSguess(object):

    def check_optts_convergence(self):

        if not optts_converged(self.optts_out_lines):
            if is_optts_nearly_converged(self.optts_out_lines):
                logger.info('OptTS nearly did converge. Will try more steps')
                self.optts_nearly_converged = True
                self.xyzs = get_orca_opt_final_xyzs(self.optts_out_lines)
                self.name = self.name + '_reopt'
                return self.run_orca_optts()

            logger.warning('OptTS calculation was no where near converging')

        else:
            self.optts_converged = True

    def do_displacements(self):
        self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_out_lines, displacement_magnitude=1.0)
        self.name += 'dis'
        self.run_orca_optts()

        imag_freqs, ts_xyzs, ts_energy = get_imag_frequencies_xyzs_energy_optts(self.optts_out_lines)
        self.check_optts_convergence()

        if len(imag_freqs) > 1:
            logger.warning('OptTS calculation returned {} imaginary frequencies'.format(len(imag_freqs)))
            self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_out_lines,
                                                                displacement_magnitude=-1.0)
            self.name += 'dis2'
            self.run_orca_optts()
            self.check_optts_convergence()

            if len(imag_freqs) > 1:
                logger.error('Couldn\'t remove other imaginary frequencies by displacement')

        return self

    def run_orca_optts(self):
        logger.info('Getting ORCA out lines from OptTS calculation')

        orca_inp_filename = self.name + '_orca_opt.inp'
        gen_orca_inp(orca_inp_filename, Config.opt_ts_keywords, self.xyzs, self.charge, self.mult, self.solvent,
                     Config.n_cores, add_bond_ids=self.bonds_to_add, optts=True)

        self.optts_out_lines = run_orca(orca_inp_filename, out_filename=orca_inp_filename.replace('.inp', '.out'))

    def get_imag_frequencies_xyzs_energy(self):
        return get_imag_frequencies_xyzs_energy_optts(self.optts_out_lines)

    def __init__(self, name='ts_guess', reaction_class=None, xyzs=None, solvent=None, charge=0, mult=1,
                 bonds_to_add=None, optts_out_lines=None):
        """

        :param name:
        :param reaction_class:
        :param xyzs:
        :param solvent:
        :param charge:
        :param mult:
        :param bonds_to_add:
        """
        self.name = name
        self.xyzs = xyzs
        self.n_atoms = len(xyzs) if xyzs is not None else None
        self.reaction_class = reaction_class
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
        self.bonds_to_add = bonds_to_add
        self.optts_out_lines = optts_out_lines

        self.optts_converged = False
        self.optts_nearly_converged = False
