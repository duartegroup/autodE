from shutil import which
import os
from autode.config import Config
import inspect
from autode.log import logger

# List of required methods that need to be added to construct a valid electronic structure method wrapper
req_methods = ['generate_input', 'calculation_terminated_normally',  'get_energy', 'optimisation_converged',
               'optimisation_nearly_converged', 'get_imag_freqs', 'get_normal_mode_displacements', 'get_final_xyzs']


class ElectronicStructureMethod:

    def set_availability(self):
        logger.info(f'Setting the availability of an electronic structure code: {self.name}')

        # Config.EST.path can be set at run time so if it's not None update self.path
        for _, est_config_class in inspect.getmembers(Config, inspect.isclass):
            if self.__name__ == est_config_class.__name__:
                if est_config_class.path is not None:
                    self.path = est_config_class.path

        if self.req_licence:
            if self.path is not None and self.path_to_licence is not None:
                if os.path.exists(self.path) and os.path.exists(self.path_to_licence):
                    self.available = True

        else:
            if self.path is not None:
                if os.path.exists(self.path):
                    self.available = True

        logger.info(f'{self.name} is available {self.available}')
        return None

    def reset(self, Config):

        attributes = ['path', 'path_to_licence', 'scan_keywords', 'conf_opt_keywords', 'opt_keywords', 'opt_ts_keywords', 'hess_keywords', 'opt_ts_block', 'sp_keywords']
        esw = getattr(Config, self.__name__)
        for attribute in attributes:
            try:
                new_attr_val = getattr(esw, attribute)
                if new_attr_val is not None:
                    setattr(self, attribute, new_attr_val)
            except(AttributeError):
                pass

    def __init__(self, name, path, req_licence=False, path_to_licence=None, aval_solvents=None, scan_keywords=None,
                 conf_opt_keywords=None, opt_keywords=None, opt_ts_keywords=None, hess_keywords=None, opt_ts_block=None,
                 sp_keywords=None, mpirun=False):
        """
        Arguments:
            name (str): wrapper name. ALSO the name of the executable
            path (str): absolute path to the executable

        Keyword Arguments:
            req_licence (bool): does the method require a licence file to run? (default: {False})
            path_to_licence (str): absolute path to the licence file if it is required (default: {None})
            aval_solvents (list): list of available solvents (default: {None})
            scan_keywords (list): keywords to use if performing a relaxed PES scan (default: {None})
            conf_opt_keywords (list): keywords to use to optimised conformers (default: {None})
            opt_keywords (list): keywords to use when performing a regular optimisation (default: {None})
            opt_ts_keywords (list): keywords to use when performing a TS search (default: {None})
            hess_keywords (list): keywords to use when just performing a hessian calculation (default: {None})
            opt_ts_block (list): additional OptTS parameters (default: {None})
            sp_keywords (list): keywords to use when performing a single point calculation (default: {None})
            mpirun (bool): does the method need mpirun to call it? (default:{False})
        """
        self.__name__ = name
        self.name = name

        # If the path is not set in config.py search in $PATH
        self.path = path if path is not None else which(name)
        self.req_licence = req_licence
        self.path_to_licence = path_to_licence
        self.aval_solvents = aval_solvents

        # Set the availability of the method
        self.available = False
        self.set_availability()

        self.scan_keywords = scan_keywords
        self.conf_opt_keywords = conf_opt_keywords
        self.opt_keywords = opt_keywords
        self.opt_ts_keywords = opt_ts_keywords
        self.hess_keywords = hess_keywords
        self.opt_ts_block = opt_ts_block
        self.sp_keywords = sp_keywords
        self.mpirun = mpirun
