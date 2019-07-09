from .log import logger
from .config import Config
from .single_point import get_single_point_energy


class TS(object):

    def single_point(self):
        self.energy = get_single_point_energy(self, keywords=Config.sp_keywords, n_cores=Config.n_cores)

    def is_true_ts(self):
        if len(self.imag_freqs) == 1:
            return True
        else:
            return False

    def __init__(self, imag_freqs=None, xyzs=None, energy=None, name='TS', solvent=None, charge=0, mult=1,
                 converged=True):
        """
        Generate a TS object
        :param imag_freqs: (list) List of imaginary frequencies given as negative value
        """
        logger.info('Generating a TS object for {}'.format(name))

        self.name = name
        self.xyzs = xyzs
        self.energy = energy
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
        self.converged = converged
        self.imag_freqs = imag_freqs
