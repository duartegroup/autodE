from abc import ABC, abstractmethod
from copy import deepcopy
from shutil import which
from typing import Optional, List
from pathlib import Path
from autode.log import logger
from autode.values import PotentialEnergy, Gradient, Coordinates
from autode.hessians import Hessian
from autode.exceptions import NotImplementedInMethod
from autode.wrappers.keywords import ImplicitSolventType, KeywordsSet
from autode.calculations.types import CalculationType as ct


class Method(ABC):
    def __init__(
        self, name: str, keywords_set: KeywordsSet, doi_list: List[str]
    ):
        """
        A base autodE method wrapper, capable of setting energies/gradients/
        Hessians of a molecule

        -----------------------------------------------------------------------
        Arguments:
            name: Name of this method

            keywords_set: Set of keywords to use for different types of
                          calculations

            doi_list: List of digital object identifiers (DOIs)
        """

        self._name = name
        self.keywords = keywords_set.copy()
        self.implicit_solvation_type = None
        self.doi_list = doi_list

    @property
    def name(self) -> str:
        """
        Name of this method. e.g. "g09" for Gaussian 09
        """
        return self._name

    def execute(self, calc: "CalculationExecutor") -> None:
        pass

    @property
    @abstractmethod
    def uses_external_io(self) -> bool:
        """
        Does this method generate an input/output file that needs to be parsed
        to find the required properties e.g. energy of the input molecule.
        """

    @abstractmethod
    def __repr__(self):
        """Representation of this method"""

    @abstractmethod
    def implements(
        self, calculation_type: "autode.calculations.types.CalculationType"
    ) -> bool:
        """Does this method implement a particular calculation type?"""

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        """Did the calculation terminate normally?"""
        return True

    @property
    def doi_str(self):
        return " ".join(self.doi_list)

    @property
    def available_implicit_solvents(self) -> List[str]:
        """Available implicit solvent models for this EST method"""
        from autode.solvent.solvents import solvents

        return [
            s.name for s in solvents if s.is_implicit and hasattr(s, self.name)
        ]

    @property
    def is_available(self):
        """Is this method available?"""
        return True

    def version_in(self, calc: "CalculationExecutor") -> str:
        """Determine the version of the method used in this calculation"""
        return "???"

    def _all_equal(self, other, attrs) -> bool:
        return all(getattr(other, a) == getattr(self, a) for a in attrs)

    def __eq__(self, other) -> bool:
        """Equality of this method to another one"""

        if not isinstance(other, self.__class__):
            return False

        return self._all_equal(other, attrs=("name", "keywords"))

    def copy(self) -> "Method":
        return deepcopy(self)


class ExternalMethod(Method, ABC):
    def __init__(
        self,
        executable_name: str,
        keywords_set: KeywordsSet,
        doi_list: List[str],
        implicit_solvation_type: ImplicitSolventType,
        path: Optional[str] = None,
    ):
        """
        An autodE wrapped method that calls an executable to generate an output
        file

        -----------------------------------------------------------------------
        Arguments:
            executable_name: Name of the executable to call e.g. orca

            implicit_solvation_type: Type of implicit solvent

            path: Full file path to the executable. Overrides the path found
                  when calling

        See Also:

            :py:meth:`Method <Method.__init__>`
        """
        super().__init__(
            name=executable_name, keywords_set=keywords_set, doi_list=doi_list
        )

        self.implicit_solvation_type = implicit_solvation_type
        self.path = path if path is not None else which(executable_name)

    @property
    def is_available(self):
        """Is this method available?"""
        logger.info(f"Setting the availability of {self.name}")

        if self.path is not None:
            if Path(self.path).exists():
                logger.info(f"{self.name} is available")
                return True

        logger.info(f"{self.name} is not available")
        return False

    @abstractmethod
    def execute(self, calc: "CalculationExecutor") -> None:
        """Run this calculation and generate an output file"""

    @abstractmethod
    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        """Did the calculation terminate normally?"""

    @abstractmethod
    def optimiser_from(
        self, calc: "CalculationExecutor"
    ) -> "autode.opt.optimisers.base.BaseOptimiser":
        """
        Optimiser that this method used. Set from the calculation output
        """

    def energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        """
        Get an energy with a set of associated attributes, defined by the
        method which was used to execute the calculation.
        """
        logger.info(f"Getting energy from {calc.output.filename}")

        energy = self._energy_from(calc)
        if energy is not None:
            energy.set_method_str(method=self, keywords=calc.input.keywords)

        return energy

    @abstractmethod
    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        """
        Extract the energy from this calculation
        """

    @abstractmethod
    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        """
        Extract the gradient from this calculation
        """

    @abstractmethod
    def hessian_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> Hessian:
        """
        Extract the Hessian from this calculation
        """

    @abstractmethod
    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        """
        Extract the final set of atomic coordinates from the output file. They
        *must* be in the same order as they were specified
        """

    def atoms_from(self, calc: "CalculationExecutor") -> "Atoms":
        """
        Extract the atoms from a completed calculation
        """

        atoms = calc.molecule.atoms.copy()
        atoms.coordinates = self.coordinates_from(calc)
        return atoms

    @abstractmethod
    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        """
        Extract the partial atomic charges corresponding to the final geometry
        present in the output file
        """

    @abstractmethod
    def version_in(self, calc: "CalculationExecutor") -> str:
        """Determine the version of the method used in this calculation"""

    @property
    def uses_external_io(self) -> bool:
        return True

    @staticmethod
    @abstractmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        """Determine the input filename for a calculation"""

    @staticmethod
    @abstractmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        """Determine the output filename for a calculation"""

    @abstractmethod
    def generate_input_for(self, calc: "CalculationExecutor") -> None:
        """Generate the input required for a calculation"""

    def __eq__(self, other) -> bool:
        attrs = ("name", "keywords", "path", "implicit_solvation_type")
        return isinstance(other, self.__class__) and self._all_equal(
            other, attrs
        )


class ExternalMethodOEG(ExternalMethod, ABC):
    """External method that implements optimisation, energy and gradient"""

    def implements(
        self, calculation_type: "autode.calculations.types.CalculationType"
    ) -> bool:
        return calculation_type in (ct.opt, ct.energy, ct.gradient)

    def hessian_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> Hessian:
        raise NotImplementedInMethod


class ExternalMethodOEGH(ExternalMethod, ABC):
    """External method that implements opt, energy, gradient and Hessians"""

    def implements(
        self, calculation_type: "autode.calculations.types.CalculationType"
    ) -> bool:
        return calculation_type in (ct.opt, ct.energy, ct.gradient, ct.hessian)


class ExternalMethodEGH(ExternalMethod, ABC):
    """External method that implements energy, gradient and Hessians"""

    def implements(
        self, calculation_type: "autode.calculations.types.CalculationType"
    ) -> bool:
        return calculation_type in (ct.energy, ct.gradient, ct.hessian)

    def optimiser_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> "autode.opt.optimisers.base.BaseOptimiser":
        raise NotImplementedInMethod
