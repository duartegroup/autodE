from shutil import which
import os
from autode.log import logger

# List of required methods that need to be added to construct a valid electronic structure method wrapper
req_methods = ['generate_input', 'calculation_terminated_normally',  'get_energy', 'optimisation_converged',
               'optimisation_nearly_converged', 'get_imag_freqs', 'get_normal_mode_displacements', 'get_final_xyzs']


class ElectronicStructureMethod:

    def set_availability(self):
        logger.info('Setting the availability of an electronic structure code')

        if self.req_licence:
            if self.path is not None and self.path_to_licence is not None:
                if os.path.exists(self.path) and os.path.exists(self.path_to_licence):
                    self.available = True

        else:
            if self.path is not None:
                if os.path.exists(self.path):
                    self.available = True

        return None

    def __init__(self, name, path, req_licence=False, path_to_licence=None, aval_solvents=None, scan_keywords=None,
                 conf_opt_keywords=None, opt_keywords=None, opt_ts_keywords=None, hess_keywords=None, opt_ts_block=None,
                 sp_keywords=None):
        """
        :param name: (str)
        :param path: (str) absolute path to the executable
        :param req_licence: (bool) does the method require a licence file to run?
        :param path_to_licence: (str) absolute path to the licence file if it is required
        :param aval_solvents: (list) list of available solvents
        :param scan_keywords: (list) keywords to use if performing a relaxed PES scan
        :param conf_opt_keywords: (list) keywords to use to optimised conformers
        :param opt_keywords: (list) keywords to use when performing a regular optimisation
        :param opt_ts_keywords: (list) keywords to use when performing a TS search
        :param hess_keywords: (list) keywords to use when just performing a hessian calculation
        :param opt_ts_block: (str) additional OptTS parameters
        :param sp_keywords: (list) keywords to use when performing a single point calculation
        """

        self.path = path if path is not None else which(name)      # If the path is not set in config.py search in $PATH
        self.req_licence = req_licence
        self.path_to_licence = path_to_licence
        self.aval_solvents = aval_solvents

        self.available = False
        self.set_availability()

        self.scan_keywords = scan_keywords
        self.conf_opt_keywords = conf_opt_keywords
        self.opt_keywords = opt_keywords
        self.opt_ts_keywords = opt_ts_keywords
        self.hess_keywords = hess_keywords
        self.opt_ts_block = opt_ts_block
        self.sp_keywords = sp_keywords
