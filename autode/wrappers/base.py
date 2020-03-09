from shutil import which
import os
import inspect
from autode.log import logger
from abc import ABC, abstractmethod


class ElectronicStructureMethod(ABC):

    def set_availability(self):
        logger.info(f'Setting the availability of {self.__name__}')

        if self.req_licence:
            if self.path is not None and self.path_to_licence is not None:
                if os.path.exists(self.path) and os.path.exists(self.path_to_licence):
                    self.available = True
                    logger.info(f'{self.__name__} is available')

        else:
            if self.path is not None:
                if os.path.exists(self.path):
                    self.available = True
                    logger.info(f'{self.__name__} is available')

        if not self.available:
            logger.info(f'{self.__name__} is not available')

    @abstractmethod
    def generate_input(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def calculation_terminated_normally(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def get_energy(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def optimisation_converged(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def optimisation_nearly_converged(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def get_imag_freqs(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def get_normal_mode_displacements(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def get_final_xyzs(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def get_atomic_charges(self):
        """
        Function implemented in individual child classes
        """
        pass

    @abstractmethod
    def get_gradients(self):
        """
        Function implemented in individual child classes
        """
        pass

    def __init__(self, name, path, req_licence=False, path_to_licence=None, scan_keywords=None,
                 conf_opt_keywords=None, gradients_keywords=None, opt_keywords=None, opt_ts_keywords=None, hess_keywords=None,
                 opt_ts_block=None, sp_keywords=None, mpirun=False):
        """
        Arguments:
            name (str): wrapper name. ALSO the name of the executable
            path (str): absolute path to the executable

        Keyword Arguments:
            req_licence (bool): does the method require a licence file to run? (default: {False})
            path_to_licence (str): absolute path to the licence file if it is required (default: {None})
            scan_keywords (list): keywords to use if performing a relaxed PES scan (default: {None})
            conf_opt_keywords (list): keywords to use to optimise conformers (default: {None})
            gradients_keywords (list): keywords to use to get the gradients (default: {None})
            opt_keywords (list): keywords to use when performing a regular optimisation (default: {None})
            opt_ts_keywords (list): keywords to use when performing a TS search (default: {None})
            hess_keywords (list): keywords to use when just performing a hessian calculation (default: {None})
            opt_ts_block (list): additional OptTS parameters (default: {None})
            sp_keywords (list): keywords to use when performing a single point calculation (default: {None})
            mpirun (bool): does the method need mpirun to call it? (default:{False})
        """
        self.name = name
        self.__name__ = self.__class__.__name__

        # If the path is not set in config.py or input script search in $PATH
        self.path = path if path is not None else which(name)
        self.req_licence = req_licence
        self.path_to_licence = path_to_licence

        # Availability is set when hlevel and llevel methods are set
        self.available = False

        self.scan_keywords = scan_keywords
        self.conf_opt_keywords = conf_opt_keywords
        self.gradients_keywords = gradients_keywords
        self.opt_keywords = opt_keywords
        self.opt_ts_keywords = opt_ts_keywords
        self.hess_keywords = hess_keywords
        self.opt_ts_block = opt_ts_block
        self.sp_keywords = sp_keywords
        self.mpirun = mpirun
