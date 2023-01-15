import autode.wrappers.keywords as kws
import autode.exceptions as ex

from copy import deepcopy
from typing import Optional, List
from autode.utils import deprecated
from autode.atoms import Atoms
from autode.point_charges import PointCharge
from autode.log import logger
from autode.hessians import Hessian
from autode.values import PotentialEnergy, Gradient
from autode.calculations.types import CalculationType
from autode.calculations.executors import (
    CalculationExecutor,
    CalculationExecutorO,
    CalculationExecutorG,
    CalculationExecutorH,
)

output_exts = (
    ".out",
    ".hess",
    ".xyz",
    ".inp",
    ".com",
    ".log",
    ".nw",
    ".pc",
    ".grad",
)


class Calculation:
    def __init__(
        self,
        name: str,
        molecule: "autode.species.Species",
        method: "autode.wrappers.methods.Method",
        keywords: "autode.wrappers.keywords.Keywords",
        n_cores: int = 1,
        point_charges: Optional[List[PointCharge]] = None,
    ):
        """
        Calculation e.g. single point energy evaluation on a molecule. This
        will update the molecule inplace. For example, an optimisation will
        alter molecule.atoms.

        -----------------------------------------------------------------------
        Arguments:
            name: Name of the calculation. Will be modified with a method
                  suffix

            molecule: Molecule to be calculated. This may have a set of
                      associated cartesian or distance constraints

            method: Wrapped electronic structure method, or other e.g.
                    forcefield capable of calculating energies and gradients

            keywords: Keywords defining the type of calculation and e.g. what
                      basis set and functional to use.

            n_cores: Number of cores available (default: {1})

            point_charges: List of  float of point charges
        """

        self.name = name
        self.n_cores = int(n_cores)
        self.point_charges = point_charges
        self._executor = self._executor_for(molecule, method, keywords)

        self._check()

    def _executor_for(
        self,
        molecule: "autode.species.Species",
        method: "autode.wrappers.methods.Method",
        keywords: "autode.wrappers.keywords.Keywords",
    ) -> "autode.calculations.executors.CalculationExecutor":
        """
        Return a calculation executor depending on the calculation modes
        implemented in the wrapped method. For instance if the method does not
        implement any optimisation then use an executor that uses the in built
        autodE optimisers (in autode/opt/). Equally if the method does not
        implement way of calculating Hessians then use a numerical evaluation
        of the Hessian
        """
        _type = CalculationExecutor  # base type, implements all calc types

        if _are_opt(keywords) and not method.implements(CalculationType.opt):
            _type = CalculationExecutorO

        if _are_grad(keywords) and not method.implements(
            CalculationType.gradient
        ):
            _type = CalculationExecutorG

        if _are_hess(keywords) and not method.implements(
            CalculationType.hessian
        ):
            _type = CalculationExecutorH

        return _type(
            self.name,
            molecule,
            method,
            keywords,
            self.n_cores,
            self.point_charges,
        )

    def run(self) -> None:
        """Run the calculation using the EST method"""
        logger.info(f"Running calculation: {self.name}")

        self._executor.run()
        self._check_properties_exist()
        self._add_to_comp_methods()

        return None

    def clean_up(self, force: bool = False, everything: bool = False) -> None:
        """
        Clean up input and output files, if Config.keep_input_files is False
        (and not force=True)

        -----------------------------------------------------------------------
        Keyword Arguments:

            force (bool): If True then override Config.keep_input_files

            everything (bool): Remove both input and output files
        """
        return self._executor.clean_up(force, everything)

    def generate_input(self) -> None:
        """Generate the input required for this calculation"""

        if not self.method.uses_external_io:
            logger.warning(
                "Calculation does not create an input file. No "
                "input has been generated"
            )
        else:
            self._executor.generate_input()

    @property
    def terminated_normally(self) -> bool:
        """
        Determine if the calculation terminated without error

        -----------------------------------------------------------------------
        Returns:
            (bool): Normal termination of the calculation?
        """
        return self._executor.terminated_normally

    @property
    def input(self) -> "autode.calculations.input.CalculationInput":
        """The input used to run this calculation"""
        return self._executor.input

    @property
    def output(self) -> "autode.calculations.output.CalculationOutput":
        """The output generated by this calculation"""
        return self._executor.output

    def set_output_filename(self, filename: str) -> None:
        """
        Set the output filename. If it exists then the properties of
        the molecule this calculation was created with from will be
        set
        """
        self._executor.output.filename = filename
        self._executor.set_properties()
        self._check_properties_exist()
        return None

    @property
    def optimiser(self) -> "autode.opt.optimisers.base.BaseOptimiser":
        """The optimiser used to run this calculation"""
        return self._executor.optimiser

    def copy(self) -> "Calculation":
        return deepcopy(self)

    @property
    def molecule(self) -> "autode.species.Species":
        return self._executor.molecule

    @molecule.setter
    def molecule(self, value: "autode.species.Species"):
        self._executor.molecule = value

    @property
    def keywords(self) -> "autode.wrappers.keywords.Keywords":
        return self._executor.input.keywords

    @property
    def method(self) -> "autode.wrappers.methods.Method":
        return self._executor.method

    def _check(self) -> None:
        """
        Ensure the molecule has the required properties and raise exceptions
        if they are not present. Also ensure that the method has the requsted
        solvent available.

        -----------------------------------------------------------------------
        Raises:
            (ValueError | SolventUnavailable | NoInputError):
        """
        from autode.species.species import Species

        assert isinstance(self.molecule, Species)

        if self.molecule.atoms is None or self.molecule.n_atoms == 0:
            raise ex.NoInputError("Have no atoms. Can't form a calculation")

        if not self.molecule.has_valid_spin_state:
            raise ex.CalculationException(
                f"Cannot execute a calculation without a valid spin state: "
                f"Spin multiplicity (2S+1) = {self.molecule.mult}"
            )

        return None

    def _add_to_comp_methods(self) -> None:
        """Add the methods used in this calculation to the used methods list"""
        from autode.log.methods import methods

        methods.add(
            f"Calculations were performed using {self.method.name} v. "
            f"{self.method.version_in(self)} "
            f"({self.method.doi_str})."
        )

        # Type of calculation ----
        if isinstance(self.input.keywords, kws.SinglePointKeywords):
            string = "Single point "

        elif isinstance(self.input.keywords, kws.OptKeywords):
            string = "Optimisation "

        else:
            logger.warning(
                "Not adding gradient or hessian to methods section "
                "anticipating that they will be the same as opt"
            )
            # and have been already added to the methods section
            return

        # Level of theory ----
        string += (
            f"calculations performed at the "
            f"{self.input.keywords.method_string} level"
        )

        basis = self.input.keywords.basis_set
        if basis is not None:
            string += (
                f" in combination with the {str(basis)} "
                f"({basis.doi_str}) basis set"
            )

        if self.molecule.solvent is not None:
            solv_type = self.method.implicit_solvation_type
            doi = solv_type.doi_str if hasattr(solv_type, "doi_str") else "?"

            string += (
                f" and {solv_type.upper()} ({doi}) "
                f"solvation, with parameters appropriate for "
                f"{self.molecule.solvent}"
            )

        methods.add(f"{string}.\n")
        return None

    def _check_properties_exist(self) -> None:
        """
        Check that the requested properties, as defined by the type of keywords
        that this calculation was requested with have been set.

        -----------------------------------------------------------------------
        Raises:
            (CouldNotGetProperty): If the required property couldn't be found
        """
        logger.info("Checking required properties exist")

        if not self.terminated_normally:
            logger.error(
                f"Calculation of {self.molecule} did not terminate "
                f"normally"
            )
            raise ex.CouldNotGetProperty()

        if self.molecule.energy is None:
            raise ex.CouldNotGetProperty(name="energy")

        if _are_grad(self.keywords) and self.molecule.gradient is None:
            raise ex.CouldNotGetProperty(name="gradient")

        if _are_hess(self.keywords) and self.molecule.hessian is None:
            raise ex.CouldNotGetProperty(name="Hessian")

        return None

    # ----------------- Deprecated functions ----------------------

    @deprecated
    def get_energy(self) -> PotentialEnergy:
        self._executor.set_properties()
        return self.molecule.energy

    @deprecated
    def optimisation_converged(self) -> bool:
        self._executor.set_properties()
        return self.optimiser.converged

    @deprecated
    def optimisation_nearly_converged(self) -> bool:
        self._executor.set_properties()
        tol = PotentialEnergy(0.1, units="kcal mol-1")
        return (
            not self.optimiser.converged
            and self.optimiser.last_energy_change < tol
        )

    @deprecated
    def get_final_atoms(self) -> Atoms:
        self._executor.set_properties()
        return self.molecule.atoms

    @deprecated
    def get_atomic_charges(self) -> List[float]:
        self._executor.set_properties()
        return self.molecule.partial_charges

    @deprecated
    def get_gradients(self) -> Gradient:
        self._executor.set_properties()
        return self.molecule.gradient

    @deprecated
    def get_hessian(self) -> Hessian:
        self._executor.set_properties()
        return self.molecule.hessian


def _are_opt(keywords) -> bool:
    return isinstance(keywords, kws.OptKeywords)


def _are_grad(keywords) -> bool:
    return isinstance(keywords, kws.GradientKeywords)


def _are_hess(keywords) -> bool:
    return isinstance(keywords, kws.HessianKeywords)
