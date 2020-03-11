from shutil import which
import os
import inspect
from autode.log import logger
from abc import ABC, abstractmethod


class ElectronicStructureMethod(ABC):

    def set_availability(self):
        logger.info(f'Setting the availability of {self.__name__}')

        if self.path is not None:
                if os.path.exists(self.path):
                    self.available = True
                    logger.info(f'{self.__name__} is available')

        if not self.available:
            logger.info(f'{self.__name__} is not available')

    @abstractmethod
    def generate_input(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def calculation_terminated_normally(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_energy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def optimisation_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def optimisation_nearly_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_imag_freqs(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_normal_mode_displacements(self, calc, mode_number):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
            mode_number (int): Number of the normal mode to ge the displacements along 6 == first imaginary mode
        """
        pass

    @abstractmethod
    def get_final_xyzs(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_atomic_charges(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_gradients(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    def __init__(self, name, path, keywords, mpirun=False):
        """
        Arguments:
            name (str): wrapper name. ALSO the name of the executable
            path (str): absolute path to the executable
            keywords (autode.wrappers.keywords.Keywords): keywords_list to use in calculations with this method

        """
        self.name = name
        self.__name__ = self.__class__.__name__

        # If the path is not set in config.py or input script search in $PATH
        self.path = path if path is not None else which(name)

        # Availability is set when hlevel and llevel methods are set
        self.available = False

        self.keywords = keywords
        self.mpirun = mpirun
