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
from autode.calculations.output import CalculationOutput


class CalculationExecutor:

    def __init__(self,
                 name:          str,
                 molecule:      'autode.species.Species',
                 method:        'autode.wrappers.base.Method',
                 keywords:      'autode.wrappers.keywords.Keywords',
                 n_cores:       int = 1,
                 point_charges: Optional[List[PointCharge]] = None):

        # Calculation names that start with "-" can break EST methods
        self.name = f'{_string_without_leading_hyphen(name)}_{method.name}'

        self.molecule = molecule
        self.method = method
        self.optimiser = NullOptimiser()
        self.n_cores = int(n_cores)

        self.input = CalculationInput(keywords=keywords.copy(),
                                      added_internals=_active_bonds(molecule),
                                      point_charges=point_charges)
        self.output = CalculationOutput()
        self._check()

    def _check(self) -> None:
        """Check that the method has the required properties to run the calc"""

        if (self.molecule.solvent is not None
                and self.molecule.solvent.is_implicit
                and not hasattr(self.molecule.solvent, self.method.name)):

            m_name = self.method.name
            err_str = (f'Could not find {self.molecule.solvent} for '
                       f'{m_name}. Available solvents for {m_name} '
                       f'are: {self.method.available_implicit_solvents}')

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
        logger.info(f'Generating input file(s) for {self.name}')

        # Can switch off uniqueness testing with e.g.
        # export AUTODE_FIXUNIQUE=False   used for testing
        if os.getenv('AUTODE_FIXUNIQUE', True) != 'False':
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
                err_str = (f'Keyword: {keyword} is not supported set '
                           f'{repr(keyword)}.{self.method.name} as a string')
                raise ex.UnsupportedCalculationInput(err_str)

        return self.method.generate_input_for(self)

    def _execute_external(self) -> None:
        """
        Execute an external calculation i.e. one that saves a log file if it
        has not been run, or if it did not finish with a normal termination
        """
        logger.info(f'Running {self.input.filename} using {self.method.name}')

        if not self.input.exists:
            raise ex.NoInputError('Input did not exist')

        if self.output.exists and self.terminated_normally:
            logger.info('Calculation already terminated normally. Skipping')
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

        self.molecule.energy = self.method.energy_from(self)

        if isinstance(keywords, kws.OptKeywords):
            self.optimiser = self.method.optimiser_from(self)
            self.molecule.coordinates = self.method.coordinates_from(self)

        if isinstance(keywords, kws.GradientKeywords):
            self.molecule.gradient = self.method.gradient_from(self)
        else:  # Try to set the gradient anyway
            self._no_except_set_gradient()

        if isinstance(keywords, kws.HessianKeywords):
            self.molecule.hessian = self.method.hessian_from(self)
        else:  # Try to set hessian anyway
            self._no_except_set_hessian()
          
        try:
            self.molecule.partial_charges = self.method.partial_charges_from(self)
        except (ValueError, IndexError, ex.AutodeException):
            logger.warning("Failed to set partial charges")

        return None
    
    @no_exceptions
    def _no_except_set_gradient(self) -> None:
        self.molecule.gradient = self.method.gradient_from(self)

    @no_exceptions
    def _no_except_set_hessian(self) -> None:
        self.molecule.hessian = self.method.hessian_from(self)

    def clean_up(self,
                 force:      bool = False,
                 everything: bool = False
                 ) -> None:

        if Config.keep_input_files and not force:
            logger.info('Keeping input files')
            return None

        filenames = self.input.filenames
        if everything:
            filenames.append(self.output.filename)
            filenames += [fn for fn in os.listdir() if fn.startswith(self.name)]

        logger.info(f'Deleting: {set(filenames)}')

        for filename in set(filenames):

            try:
                os.remove(filename)
            except FileNotFoundError:
                logger.warning(f'Could not delete {filename} it did not exist')

        return None

    @property
    def terminated_normally(self) -> bool:
        """
        Determine if the calculation terminated without error

        -----------------------------------------------------------------------
        Returns:
            (bool): Normal termination of the calculation?
        """
        logger.info(f'Checking for {self.output.filename} normal termination')

        if not self.output.exists:
            logger.warning('Calculation did not generate any output')
            return False

        return self.method.terminated_normally_in(self)

    def copy(self) -> "CalculationExecutor":
        return deepcopy(self)

    def __str__(self):
        """Create a unique string(/hash) of the calculation"""
        string = (f'{self.name}{self.method.name}{repr(self.input.keywords)}'
                  f'{self.molecule}{self.method.implicit_solvation_type}'
                  f'{self.molecule.constraints}')

        hasher = hashlib.sha1(string.encode()).digest()
        return base64.urlsafe_b64encode(hasher).decode()

    def _fix_unique(self, register_name='.autode_calculations') -> None:
        """
        If a calculation has already been run for this molecule then it
        shouldn't be run again, unless the input keywords have changed, in
        which case it should be run while retaining the previous data. This
        function fixes this problem by checking .autode_calculations and adding
        a number to the end of self.name if the calculation input is different
        """
        def append_register():
            with open(register_name, 'a') as register_file:
                print(self.name, str(self), file=register_file)

        def exists():
            return any(reg_name == self.name for reg_name in register.keys())

        def is_identical():
            return any(reg_id == str(self) for reg_id in register.values())

        # If there is no register yet in this folder then create it
        if not os.path.exists(register_name):
            logger.info('No calculations have been performed here yet')
            append_register()
            return None

        # Populate a register of calculation names and their unique identifiers
        register = {}
        for line in open(register_name, 'r'):
            if len(line.split()) == 2:                  # Expecting: name id
                calc_name, identifier = line.split()
                register[calc_name] = identifier

        if is_identical():
            logger.info('Calculation exists in registry')
            return None

        # If this calculation doesn't yet appear in the register add it
        if not exists():
            logger.info('This calculation has not yet been run')
            append_register()
            return None

        # If we're here then this calculation - with these input - has not yet
        # been run. Therefore, add an integer to the calculation name until
        # either the calculation has been run before and is the same or it's
        # not been run
        logger.info('Calculation with this name has been run before but '
                    'with different input')
        name, n = self.name, 0
        while True:
            self.name = f'{name}{n}'
            logger.info(f'New calculation name is: {self.name}')

            if is_identical():
                return None

            if not exists():
                append_register()
                return

            n += 1


class CalculationExecutorO(CalculationExecutor):
    """Calculation executor that uses autodE inbuilt optimisation"""

    def run(self) -> None:
        from autode.opt.optimisers.crfo import CRFOptimiser
        from autode.opt.optimisers.prfo import PRFOptimiser
        type_ = CRFOptimiser
        
        if self._calc_is_ts_opt:
            type_ = PRFOptimiser

        self.method.optimiser = type_(
            init_alpha=0.1,
            maxiter=self._max_opt_cycles,
            etol=3E-5,
            gtol=1E-3, # TODO: A better number here
            callback=self._save_xyz_trj,
            callback_kwargs={"molecule": self.molecule, "name": self.name}
        )

        self.method.optimiser.run(species=self.molecule,
                                  method=self.method,
                                  n_cores=self.n_cores)

    @property
    def _calc_is_ts_opt(self) -> bool:
        """Does this calculation correspond to a transition state opt"""
        return any(k.lower() == "ts" for k in self.input.keywords)

    @property
    def _max_opt_cycles(self) -> int:
        """Get the maximum num of optimisation cycles for this calculation"""
        try:
            return next(int(kwd) for kwd in self.input.keywords 
                    if isinstance(kwd, kws.MaxOptCycles))
        except StopIteration:
            return 30

    @staticmethod
    def _save_xyz_trj(coords:   "OptCoordinates",
                      molecule: "Species",
                      name:      str
                      ) -> None:
        """Save the trajectory to a file"""

        tmp_mol = molecule.new_species()
        tmp_mol.print_xyz_file(title_line=f"E = {coords.e} Ha",
                               filename=f"{name}_opt.xyz",
                               append=True)
        return None


class CalculationExecutorG(CalculationExecutor):
    """Calculation executor with a numerical gradient evaluation"""

    def run(self) -> None:
        raise NotImplementedError


class CalculationExecutorH(CalculationExecutor):
    """Calculation executor with a numerical Hessian evaluation"""

    def run(self) -> None:
        logger.warning(f'{self.method} does not implement Hessian '
                       f'calculations. Evaluating a numerical Hessian')

        from autode.hessians import NumericalHessianCalculator

        nhc = NumericalHessianCalculator(
            self,
            method=self.method,
            keywords=kws.GradientKeywords(self.input.keywords.tolist()),
            do_c_diff=False,
            shift=Distance(2E-3, units='Å'),
            n_cores=self.n_cores
        )
        nhc.calculate()
        self.molecule.hessian = nhc.hessian


def _string_without_leading_hyphen(s: str) -> str:
    return s if not s.startswith('-') else f'_{s}'


def _active_bonds(molecule: 'autode.species.Species') -> List[Tuple[int, int]]:
    return [] if molecule.graph is None else molecule.graph.active_bonds
