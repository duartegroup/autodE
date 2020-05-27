from abc import ABC
from abc import abstractmethod
import os
from subprocess import Popen
from shutil import which
from autode.log import logger
from autode.utils import requires_output


def execute(calc, params):
    """Standard method to run a EST calculation"""

    with open(calc.output.filename, 'w') as output_file:
        # /path/to/method input_filename > output_filename
        subprocess = Popen(params,
                           stdout=output_file,
                           stderr=open(os.devnull, 'w'))
        subprocess.wait()

    return None


class ElectronicStructureMethod(ABC):

    def set_availability(self):
        logger.info(f'Setting the availability of {self.__name__}')

        if self.path is not None:
            if os.path.exists(self.path):
                self.available = True
                logger.info(f'{self.__name__} is available')

        if not self.available:
            logger.info(f'{self.__name__} is not available')
            self.available = False

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
    def clean_up(self, calc):
        """
        Function implemented in individual child classes

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

    def __init__(self, name, path, keywords_set):
        """
        Arguments:
            name (str): wrapper name. ALSO the name of the executable
            path (str): absolute path to the executable
            keywords_set (autode.wrappers.keywords.KeywordsSet):

        """
        self.name = name
        self.__name__ = self.__class__.__name__

        # If the path is not set in config.py or input script search in $PATH
        self.path = path if path is not None else which(name)

        # Availability is set when hlevel and llevel methods are set
        self.available = False

        self.keywords = keywords_set
