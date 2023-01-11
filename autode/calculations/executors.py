"""
A collection of calculation executors which can execute the correct set of
steps to run a calculation for a specific method, depending on what it
implements
"""
import os
import hashlib
import base64
import autode.exceptions as ex
import autode.wrappers.keywords as kws

from typing import Optional, List, Tuple
from copy import deepcopy
from autode.log import logger
from autode.config import Config
from autode.values import Distance
from autode.utils import no_exceptions, requires_output_to_exist
from autode.point_charges import PointCharge
from autode.opt.optimisers.base import NullOptimiser
from autode.calculations.input import CalculationInput
from autode.calculations.output import (
    CalculationOutput,
    BlankCalculationOutput,
)
from autode.values import PotentialEnergy, GradientRMS


class CalculationExecutor:
    def __init__(
        self,
        name: str,
        molecule: "autode.species.Species",
        method: "autode.wrappers.methods.Method",
        keywords: "autode.wrappers.keywords.Keywords",
        n_cores: int = 1,
        point_charges: Optional[List[PointCharge]] = None,
    ):

        # Calculation names that start with "-" can break EST methods
        self.name = f"{_string_without_leading_hyphen(name)}_{method.name}"

        self.molecule = molecule
        self.method = method
        self.optimiser = NullOptimiser()
        self.n_cores = int(n_cores)

        self.input = CalculationInput(
            keywords=keywords,
            added_internals=_active_bonds(molecule),
            point_charges=point_charges,
        )
        self._external_output = CalculationOutput()
        self._check()

    def _check(self) -> None:
        """Check that the method has the required properties to run the calc"""

        if (
            self.molecule.solvent is not None
            and self.molecule.solvent.is_implicit
            and not hasattr(self.molecule.solvent, self.method.name)
        ):

            m_name = self.method.name
            err_str = (
                f"Could not find {self.molecule.solvent} for "
                f"{m_name}. Available solvents for {m_name} "
                f"are: {self.method.available_implicit_solvents}"
            )

            raise ex.SolventUnavailable(err_str)

        return None

    def run(self) -> None:
        """Run/execute the calculation"""

        if self.method.uses_external_io:
            self.generate_input()
            self.output.filename = self.method.output_filename_for(self)
            self._execute_external()
            self.set_properties()
            self.clean_up()

        else:
            self.method.execute(self)

        return None

    def generate_input(self) -> None:
        """Generate the required input file"""
        logger.info(f"Generating input file(s) for {self.name}")

        # Can switch off uniqueness testing with e.g.
        # export AUTODE_FIXUNIQUE=False   used for testing
        if os.getenv("AUTODE_FIXUNIQUE", True) != "False":
            self._fix_unique()

        self.input.filename = self.method.input_filename_for(self)

        # Check that if the keyword is a autode.wrappers.keywords.Keyword then
        # it has the required name in the method used for this calculation
        for keyword in self.input.keywords:
            if not isinstance(keyword, kws.Keyword):  # allow string keywords
                continue

            # Allow for the unambiguous setting of a keyword with only a name
            if keyword.has_only_name:
                # set e.g. keyword.orca = 'b3lyp'
                setattr(keyword, self.method.name, keyword.name)
                continue

            # For a keyword e.g. Keyword(name='pbe', orca='PBE') then the
            # definition in this method is not obvious, so raise an exception
            if not hasattr(keyword, self.method.name):
                err_str = (
                    f"Keyword: {keyword} is not supported set "
                    f"{repr(keyword)}.{self.method.name} as a string"
                )
                raise ex.UnsupportedCalculationInput(err_str)

        return self.method.generate_input_for(self)

    def _execute_external(self) -> None:
        """
        Execute an external calculation i.e. one that saves a log file if it
        has not been run, or if it did not finish with a normal termination
        """
        logger.info(f"Running {self.input.filename} using {self.method.name}")

        if not self.input.exists:
            raise ex.NoInputError("Input did not exist")

        if self.output.exists and self.terminated_normally:
            logger.info("Calculation already terminated normally. Skipping")
            return None

        if not self.method.is_available:
            raise ex.MethodUnavailable(f"{self.method} was not available")

        self.output.clear()
        self.method.execute(self)

        return None

    @requires_output_to_exist
    def set_properties(self) -> None:
        """Set the properties of a molecule from this calculation"""
        keywords = self.input.keywords

        if isinstance(keywords, kws.OptKeywords):
            self.optimiser = self.method.optimiser_from(self)
            self.molecule.coordinates = self.method.coordinates_from(self)

        self.molecule.energy = self.method.energy_from(self)

        if isinstance(keywords, kws.GradientKeywords):
            self.molecule.gradient = self.method.gradient_from(self)
        else:  # Try to set the gradient anyway
            self._no_except_set_gradient()

        if isinstance(keywords, kws.HessianKeywords):
            self.molecule.hessian = self.method.hessian_from(self)
        else:  # Try to set hessian anyway
            self._no_except_set_hessian()

        try:
            self.molecule.partial_charges = self.method.partial_charges_from(
                self
            )
        except (ValueError, IndexError, ex.AutodeException):
            logger.warning("Failed to set partial charges")

        return None

    @no_exceptions
    def _no_except_set_gradient(self) -> None:
        self.molecule.gradient = self.method.gradient_from(self)

    @no_exceptions
    def _no_except_set_hessian(self) -> None:
        self.molecule.hessian = self.method.hessian_from(self)

    def clean_up(self, force: bool = False, everything: bool = False) -> None:

        if not self.method.uses_external_io:  # Then there are no i/o files
            return None

        if Config.keep_input_files and not force:
            logger.info("Keeping input files")
            return None

        filenames = self.input.filenames
        if everything:
            filenames.append(self.output.filename)
            filenames += [
                fn for fn in os.listdir() if fn.startswith(self.name)
            ]

        logger.info(f"Deleting: {set(filenames)}")

        for filename in [fn for fn in set(filenames) if fn is not None]:

            try:
                os.remove(filename)
            except FileNotFoundError:
                logger.warning(f"Could not delete {filename} it did not exist")

        return None

    @property
    def terminated_normally(self) -> bool:
        """
        Determine if the calculation terminated without error

        -----------------------------------------------------------------------
        Returns:
            (bool): Normal termination of the calculation?
        """
        logger.info(f"Checking for {self.output.filename} normal termination")

        if self.method.uses_external_io and not self.output.exists:
            logger.warning("Calculation did not generate any output")
            return False

        return self.method.terminated_normally_in(self)

    @property
    def output(self) -> "CalculationOutput":
        """
        Calculation output. If the method does not use any external files
        then a blank calculation output is returned
        """

        if self.method.uses_external_io:
            return self._external_output
        else:
            return BlankCalculationOutput()

    @output.setter
    def output(self, value: CalculationOutput):
        """Set the value of the calculation output"""
        assert isinstance(value, CalculationOutput)

        self._external_output = value

    def copy(self) -> "CalculationExecutor":
        return deepcopy(self)

    def __str__(self):
        """Create a unique string(/hash) of the calculation"""
        string = (
            f"{self.name}{self.method.name}{repr(self.input.keywords)}"
            f"{self.molecule}{self.method.implicit_solvation_type}"
            f"{self.molecule.constraints}"
        )

        hasher = hashlib.sha1(string.encode()).digest()
        return base64.urlsafe_b64encode(hasher).decode()

    def _fix_unique(self, register_name=".autode_calculations") -> None:
        """
        If a calculation has already been run for this molecule then it
        shouldn't be run again, unless the input keywords have changed, in
        which case it should be run while retaining the previous data. This
        function fixes this problem by checking .autode_calculations and adding
        a number to the end of self.name if the calculation input is different
        """

        def append_register():
            with open(register_name, "a") as register_file:
                print(self.name, str(self), file=register_file)

        def exists():
            return any(reg_name == self.name for reg_name in register.keys())

        def is_identical():
            return any(reg_id == str(self) for reg_id in register.values())

        # If there is no register yet in this folder then create it
        if not os.path.exists(register_name):
            logger.info("No calculations have been performed here yet")
            append_register()
            return None

        # Populate a register of calculation names and their unique identifiers
        register = {}
        for line in open(register_name, "r"):
            if len(line.split()) == 2:  # Expecting: name id
                calc_name, identifier = line.split()
                register[calc_name] = identifier

        if is_identical():
            logger.info("Calculation exists in registry")
            return None

        # If this calculation doesn't yet appear in the register add it
        if not exists():
            logger.info("This calculation has not yet been run")
            append_register()
            return None

        # If we're here then this calculation - with these input - has not yet
        # been run. Therefore, add an integer to the calculation name until
        # either the calculation has been run before and is the same or it's
        # not been run
        logger.info(
            "Calculation with this name has been run before but "
            "with different input"
        )
        name, n = self.name, 0
        while True:
            self.name = f"{name}{n}"
            logger.info(f"New calculation name is: {self.name}")

            if is_identical():
                return None

            if not exists():
                append_register()
                return None

            n += 1


class _IndirectCalculationExecutor(CalculationExecutor):
    """
    An 'indirect' executor is one that, given a calculation to perform,
    calls the method multiple time and aggregates the results in some way.
    Therefore, there is no direct calculation output.
    """

    @property
    def output(self) -> "CalculationOutput":
        return BlankCalculationOutput()


class CalculationExecutorO(_IndirectCalculationExecutor):
    """Calculation executor that uses autodE inbuilt optimisation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.etol = PotentialEnergy(3e-5, units="Ha")
        self.gtol = GradientRMS(
            1e-3, units="Ha Å^-1"
        )  # TODO: A better number here
        self._fix_unique()

    def run(self) -> None:
        """Run an optimisation with using default autodE optimisers"""
        from autode.opt.optimisers.crfo import CRFOptimiser
        from autode.opt.optimisers.prfo import PRFOptimiser

        if self._opt_trajectory_exists:
            self.optimiser = CRFOptimiser.from_file(self._opt_trajectory_name)
            self._set_properties_from_optimiser()
            return None

        type_ = PRFOptimiser if self._calc_is_ts_opt else CRFOptimiser

        self.optimiser = type_(
            init_alpha=self._step_size,
            maxiter=self._max_opt_cycles,
            etol=self.etol,
            gtol=self.gtol,
        )
        method = self.method.copy()
        method.keywords.grad = self.input.keywords

        self.optimiser.run(
            species=self.molecule, method=method, n_cores=self.n_cores
        )

        self.optimiser.save(self._opt_trajectory_name)

        if self.molecule.n_atoms == 1:
            return self._run_single_energy_evaluation()

        if self._calc_is_ts_opt:
            # If this calculation is a transition state optimisation then a
            # hessian on the final structure is required
            self.molecule.calc_hessian(
                method=self.method, n_cores=self.n_cores
            )
        return None

    def _run_single_energy_evaluation(self) -> None:
        """Run a single point energy evaluation, suitable for a single atom"""
        from autode.calculations.calculation import Calculation

        calc = Calculation(
            name=f"{self.molecule.name}_energy",
            molecule=self.molecule,
            method=self.method,
            keywords=kws.SinglePointKeywords(self.input.keywords),
            n_cores=self.n_cores,
        )
        calc.run()
        return None

    @property
    def terminated_normally(self) -> bool:
        """
        Using inbuilt optimisers raise exceptions if something goes wrong, so
        this property is always true, provided the output exists

        -----------------------------------------------------------------------
        Returns:
            (bool): Normal termination of the calculation?
        """
        return self._opt_trajectory_exists or self.molecule.n_atoms == 1

    def set_properties(self) -> None:
        """
        Nothing needs to be set as the energy/gradient/Hessian of the
        molecule are set within the optimiser
        """
        return None

    @property
    def _calc_is_ts_opt(self) -> bool:
        """Does this calculation correspond to a transition state opt"""
        return isinstance(self.input.keywords, kws.OptTSKeywords)

    @property
    def _max_opt_cycles(self) -> int:
        """Get the maximum num of optimisation cycles for this calculation"""
        try:
            return next(
                int(kwd)
                for kwd in self.input.keywords
                if isinstance(kwd, kws.MaxOptCycles)
            )
        except StopIteration:
            return 50

    @property
    def _step_size(self) -> float:
        return 0.05 if self._calc_is_ts_opt else 0.1

    @property
    def _opt_trajectory_name(self) -> str:
        return f"{self.name}_opt_trj.xyz"

    @property
    def _opt_trajectory_exists(self) -> bool:
        return os.path.exists(self._opt_trajectory_name)

    def _set_properties_from_optimiser(self) -> None:
        """Set the properties from the trajectory file, that must exist"""
        logger.info(
            "Setting optimised coordinates, gradient and energy from "
            "the reloaded optimiser state"
        )

        final_coords = self.optimiser.final_coordinates

        self.molecule.coordinates = final_coords.reshape((-1, 3))
        self.molecule.gradient = final_coords.g.reshape((-1, 3))
        self.molecule.energy = final_coords.e
        return None


class CalculationExecutorG(_IndirectCalculationExecutor):
    """Calculation executor with a numerical gradient evaluation"""

    def run(self) -> None:
        raise NotImplementedError


class CalculationExecutorH(_IndirectCalculationExecutor):
    """Calculation executor with a numerical Hessian evaluation"""

    def run(self) -> None:
        logger.warning(
            f"{self.method} does not implement Hessian "
            f"calculations. Evaluating a numerical Hessian"
        )

        from autode.hessians import NumericalHessianCalculator

        nhc = NumericalHessianCalculator(
            species=self.molecule,
            method=self.method,
            keywords=kws.GradientKeywords(self.input.keywords.tolist()),
            do_c_diff=False,
            shift=Distance(2e-3, units="Å"),
            n_cores=self.n_cores,
        )
        nhc.calculate()
        self.molecule.hessian = nhc.hessian

    @property
    def terminated_normally(self) -> bool:
        """
        This calculation executor terminated normally if the Hessian exists and
        did not raise any exceptions along the way

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return self.molecule.hessian is not None


def _string_without_leading_hyphen(s: str) -> str:
    return s if not s.startswith("-") else f"_{s}"


def _active_bonds(molecule: "autode.species.Species") -> List[Tuple[int, int]]:
    return [] if molecule.graph is None else molecule.graph.active_bonds
