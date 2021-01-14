from abc import ABC
from abc import abstractmethod
from shutil import which
from autode.log import logger
from autode.utils import requires_output
from copy import deepcopy
import os


class ElectronicStructureMethod(ABC):

    @property
    def available(self):
        logger.info(f'Setting the availability of {self.__name__}')

        if self.path is not None:
            if os.path.exists(self.path):
                logger.info(f'{self.__name__} is available')
                return True

        logger.info(f'{self.__name__} is not available')
        return False

    @abstractmethod
    def generate_input(self, calculation, molecule):
        """
        Function implemented in individual child classes

        Arguments:
            calculation (autode.calculation.Calculation):
            molecule (any):
        """
        pass

    def generate_explicitly_solvated_input(self, calculation_input):
        """
        Function implemented in individual child classes

        Arguments:
            calculation_input (autode.calculation.CalculationInput):
        """
        raise NotImplementedError

    def clean_up(self, calc):
        """
        Remove any input files

        Arguments:
            calc (autode.calculation.Calculation):
        """
        for filename in calc.input.get_input_filenames():
            if os.path.exists(filename):
                os.remove(filename)

        return None

    @abstractmethod
    def get_output_filename(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_input_filename(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_version(self, calc):
        """
        Get the version of the method e.g. ORCA v. 4.2.1. Function implemented
        in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def execute(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def calculation_terminated_normally(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_energy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_free_energy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_enthalpy(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def optimisation_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def optimisation_nearly_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_imaginary_freqs(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_normal_mode_displacements(self, calc, mode_number):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
            mode_number (int): Number of the normal mode to get the
            displacements along 6 == first imaginary mode
        """
        pass

    @abstractmethod
    @requires_output()
    def get_final_atoms(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_atomic_charges(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_gradients(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    def doi_str(self):
        return " ".join(self.doi_list)

    def __init__(self, name, path, keywords_set, implicit_solvation_type,
                 doi=None, doi_list=None):
        """
        Arguments:
            name (str): wrapper name. ALSO the name of the executable
            path (str): absolute path to the executable
            keywords_set (autode.wrappers.keywords.KeywordsSet):
            implicit_solvation_type (autode.wrappers.
                                     keywords.ImplicitSolventType):

        """
        self.name = name
        self.__name__ = self.__class__.__name__

        # Digital object identifier(s) of the method/or paper describing the
        # method
        self.doi_list = []
        if doi_list is not None:
            self.doi_list += doi_list

        if doi is not None:
            self.doi_list.append(doi)

        # If the path is not set in config.py or input script search in $PATH
        self.path = path if path is not None else which(name)
        self.keywords = deepcopy(keywords_set)
        self.implicit_solvation_type = implicit_solvation_type
