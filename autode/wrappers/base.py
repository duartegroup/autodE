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
        """Is this method available?"""
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
        Generate any input files required

        Arguments:
            calculation (autode.calculation.Calculation):
            molecule (any):
        """
        pass

    @abstractmethod
    def get_output_filename(self, calc):
        """
        Get the output filename from the calculation name e.g. filename.out

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    def get_input_filename(self, calc):
        """
        Get the input filename from the calculation name e.g. filename.inp

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
        Execute the calculation

        Arguments:
            calc (autode.calculation.Calculation):
        """
        pass

    @abstractmethod
    @requires_output()
    def calculation_terminated_normally(self, calc):
        """
        Did the calculation terminate correctly?

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (bool):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_energy(self, calc):
        """
        Return the potential energy

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (float | None):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_free_energy(self, calc):
        """
        Return the free energy (G)

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (float | None):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_enthalpy(self, calc):
        """
        Return the free energy (enthalpy)

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (float | None):
        """
        pass

    @abstractmethod
    @requires_output()
    def optimisation_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (bool):
        """
        pass

    @abstractmethod
    @requires_output()
    def optimisation_nearly_converged(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (bool):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_imaginary_freqs(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (list(float)):
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

        Returns:
            (np.ndarray):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_final_atoms(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (list(autode.atoms.Atom)):

        Raises:
            (autode.exceptions.AtomsNotFound)
        """
        pass

    @abstractmethod
    @requires_output()
    def get_atomic_charges(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (list(float)):
        """
        pass

    @abstractmethod
    @requires_output()
    def get_gradients(self, calc):
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (np.ndarray):

        Raisese:
            (autode.exceptions.CouldNotGetProperty)
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
