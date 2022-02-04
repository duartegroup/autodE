import os
import numpy as np
from abc import ABC
from abc import abstractmethod
from typing import List
from shutil import which
from autode.log import logger
from autode.utils import requires_output
from copy import deepcopy


class Method:
    """Base class for all EST and other energy/force/hessian methods"""

    name = 'unknown'


class ElectronicStructureMethod(Method, ABC):

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

    @abstractmethod
    def __repr__(self):
        """Representation of this method"""

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

    @property
    def available_implicit_solvents(self) -> List[str]:
        """Available implicit solvent models for this EST method"""
        from autode.solvent.solvents import solvents

        return [s.name for s in solvents
                if s.is_implicit and hasattr(s, self.name)]

    @property
    def doi_str(self):
        return " ".join(self.doi_list)

    @property
    def implements_hessian(self) -> bool:
        """
        Is either an analytic or numerical Hessian evaluation implemented
        within the electronic structure package?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return len(self.keywords.hess) > 0

    @abstractmethod
    def generate_input(self, calc, molecule):
        """
        Generate any input files required

        Arguments:
            calc (autode.calculation.Calculation):

            molecule (autode.species.Species):
        """

    @abstractmethod
    def get_output_filename(self, calc) -> str:
        """
        Get the output filename from the calculation name e.g. filename.out

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (str): Name of the output file
        """

    @abstractmethod
    def get_input_filename(self, calc) -> str:
        """
        Get the input filename from the calculation name e.g. filename.inp

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (str): Name of the input file
        """

    @abstractmethod
    def get_version(self, calc) -> str:
        """
        Get the version of the method e.g. ORCA v. 4.2.1. Function implemented
        in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (str): Version of the electronic structure method
        """

    @abstractmethod
    def execute(self, calc) -> None:
        """
        Execute the calculation

        Arguments:
            calc (autode.calculation.Calculation):
        """

    @abstractmethod
    @requires_output
    def calculation_terminated_normally(self, calc) -> bool:
        """
        Did the calculation terminate correctly?

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (bool):
        """

    @abstractmethod
    @requires_output
    def optimisation_converged(self, calc) -> bool:
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (bool):
        """

    @abstractmethod
    @requires_output
    def optimisation_nearly_converged(self, calc) -> bool:
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (bool):
        """

    @abstractmethod
    @requires_output
    def get_final_atoms(self, calc) -> List:
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (list(autode.atoms.Atom)):

        Raises:
            (autode.exceptions.AtomsNotFound)
        """

    @requires_output
    def get_atomic_charges(self, calc) -> List:
        """
        Function implemented in individual child classes

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (list(float)):
        """
        raise NotImplementedError

    @abstractmethod
    @requires_output
    def get_energy(self, calc) -> float:
        """
        Return the potential energy in electronic Hartrees

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (float):

        Raises:
            (autode.exceptions.CouldNotGetProperty)
        """

    @abstractmethod
    @requires_output
    def get_gradients(self, calc) -> np.ndarray:
        """
        Return the gradient matrix n_atomsx3 in Ha Å^-1

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (np.ndarray):

        Raises:
            (autode.exceptions.CouldNotGetProperty)
        """

    @requires_output
    def get_hessian(self, calc) -> np.ndarray:
        """
        Return the Hessian matrix 3Nx3N in Ha Å^-2 for N atoms

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (np.ndarray):

        Raises:
            (autode.exceptions.CouldNotGetProperty |
            AssertionError |
            ValueError |
            IndexError)
        """
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        """Equality of this EST method to another one"""

        if not isinstance(other, self.__class__):
            return False

        attrs = ('name', 'keywords', 'path', 'implicit_solvation_type')
        return all(getattr(other, a) == getattr(self, a) for a in attrs)
