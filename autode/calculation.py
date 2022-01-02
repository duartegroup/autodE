from copy import deepcopy
import os
import hashlib
import base64
from typing import Optional, List
import autode.wrappers.keywords as kws
import autode.exceptions as ex
from autode.utils import cached_property
from autode.atoms import Atoms
from autode.point_charges import PointCharge
from autode.constraints import Constraints
from autode.config import Config
from autode.log import logger
from autode.hessians import Hessian
from autode.values import PotentialEnergy, Gradient

output_exts = ('.out', '.hess', '.xyz', '.inp', '.com', '.log', '.nw',
               '.pc', '.grad')


def execute_calc(calc):
    """ Top level function that can be hashed"""
    return calc.execute_calculation()


class Calculation:

    def __init__(self,
                 name:                  str,
                 molecule:              'autode.species.Species',
                 method:                'autode.wrappers.base.Method',
                 keywords:              'autode.wrappers.keywords.Keywords',
                 n_cores:               int = 1,
                 bond_ids_to_add:       Optional[List[tuple]] = None,
                 other_input_block:     Optional[str] = None,
                 distance_constraints:  Optional[dict] = None,
                 cartesian_constraints: Optional[List[int]] = None,
                 point_charges:         Optional[List[PointCharge]] = None):
        """
        Arguments:
            name (str):

            molecule (autode.species.Species): Molecule to be calculated

            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (autode.wrappers.keywords.Keywords):

        Keyword Arguments:

            n_cores (int): Number of cores available (default: {1})

            bond_ids_to_add (list(tuples)): List of bonds to add to internal
                                            coordinates (default: {None})

            other_input_block (str): Other input block to add (default: {None})

            distance_constraints (dict): keys = tuple of atom ids for a bond to
                                         be kept at fixed length, value = dist
                                         to be fixed at (default: {None})

            cartesian_constraints (list(int)): List of atom ids to fix at their
                                               cartesian coordinates
                                               (default: {None})

            point_charges (list(autode.point_charges.PointCharge)): List of
                                             float of point charges, x, y, z
                                             coordinates for each point charge
        """
        # Calculation names that start with "-" can break EST methods
        self.name = (f'{name}_{method.name}' if not name.startswith('-')
                     else f'_{name}_{method.name}')

        # ------------------- System specific parameters ----------------------
        self.molecule = deepcopy(molecule)

        if hasattr(self.molecule, 'constraints'):
            self.molecule.constraints.update(distance=distance_constraints,
                                             cartesian=cartesian_constraints)
        else:
            self.molecule.constraints = Constraints(distance=distance_constraints,
                                                    cartesian=cartesian_constraints)

        # --------------------- Calculation parameters ------------------------
        self.method = method
        self.n_cores = int(n_cores)

        # ------------------- Calculation input/output ------------------------
        self.input = CalculationInput(keywords=deepcopy(keywords),
                                      additional_input=other_input_block,
                                      added_internals=bond_ids_to_add,
                                      point_charges=point_charges)

        self.output = CalculationOutput()

        self._check_molecule()

    def __str__(self):
        """Create a unique string(/hash) of the calculation"""
        string = (f'{self.name}{self.method.name}{repr(self.input.keywords)}'
                  f'{self.molecule}{self.method.implicit_solvation_type}'
                  f'{self.molecule.constraints}')

        hasher = hashlib.sha1(string.encode()).digest()
        return base64.urlsafe_b64encode(hasher).decode()

    def _check_molecule(self) -> None:
        """
        Ensure the molecule has the required properties and raise exceptions
        if they are not present.

        Raises:
            (ValueError | SolventUnavailable | NoInputError):
        """

        for attr in ('n_atoms', 'atoms', 'mult', 'charge', 'solvent'):
            if not hasattr(self.molecule, attr):
                raise ValueError(f'Molecule {self.molecule} must have '
                                 f'{attr} but was not present')

        if self.molecule.atoms is None or self.molecule.n_atoms == 0:
            raise ex.NoInputError('Have no atoms. Can\'t form a calculation')

        # Assume all calculations can be performed in the gas phase but
        # not all implicit solvents are available in all EST codes
        if (self.molecule.solvent is not None
            and self.molecule.solvent.is_implicit
            and not hasattr(self.molecule.solvent, self.method.name)):

            m_name = self.method.name
            err_str = (f'Could not find {self.molecule.solvent} for '
                       f'{m_name}. Available solvents for {m_name} '
                       f'are: {self.method.available_implicit_solvents}')

            raise ex.SolventUnavailable(err_str)

        return None

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
            return

        # Populate a register of calculation names and their unique identifiers
        register = {}
        for line in open(register_name, 'r'):
            if len(line.split()) == 2:                  # Expecting: name id
                calc_name, identifier = line.split()
                register[calc_name] = identifier

        if is_identical():
            logger.info('Calculation has already been run')
            return

        # If this calculation doesn't yet appear in the register add it
        if not exists():
            logger.info('This calculation has not yet been run')
            append_register()
            return

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
                return

            if not exists():
                append_register()
                return

            n += 1

    def _add_to_comp_methods(self) -> None:
        """Add the methods used in this calculation to the used methods list"""
        from autode.log.methods import methods

        methods.add(f'Calculations were performed using {self.method.name} v. '
                    f'{self.method.get_version(self)} '
                    f'({self.method.doi_str}).')

        # Type of calculation ----
        if isinstance(self.input.keywords, kws.SinglePointKeywords):
            string = 'Single point '

        elif isinstance(self.input.keywords, kws.OptKeywords):
            string = 'Optimisation '

        else:
            logger.warning('Not adding gradient or hessian to methods section '
                           'anticipating that they will be the same as opt')
            # and have been already added to the methods section
            return

        # Level of theory ----
        string += (f'calculations performed at the '
                   f'{self.input.keywords.method_string} level')

        basis = self.input.keywords.basis_set
        if basis is not None:
            string += (f' in combination with the {str(basis)} '
                       f'({basis.doi_str}) basis set')

        if self.molecule.solvent is not None:
            solv_type = self.method.implicit_solvation_type
            doi = solv_type.doi_str if hasattr(solv_type, 'doi_str') else '?'

            string += (f' and {solv_type.upper()} ({doi}) '
                       f'solvation, with parameters appropriate for '
                       f'{self.molecule.solvent}')

        methods.add(f'{string}.\n')
        return None

    def get_energy(self) -> Optional[PotentialEnergy]:
        """
        Total electronic potential energy

        Returns:
            (autode.values.PotentialEnergy | None):
        """
        logger.info(f'Getting energy from {self.output.filename}')

        if not self.terminated_normally:
            logger.error('Calculation did not terminate normally. '
                         'Energy = None')
            return None

        try:
            return PotentialEnergy(self.method.get_energy(self),
                                   method=self.method,
                                   keywords=self.input.keywords)

        except ex.CouldNotGetProperty:
            logger.warning('Could not get energy. Energy = None')
            return None

    def optimisation_converged(self) -> bool:
        """
        Check whether a calculation has has converged to within the theshold
        on energies and graidents specified in the input

        Returns:
            (bool):
        """
        logger.info('Checking to see if the geometry converged')
        if not self.output.exists:
            return False

        return self.method.optimisation_converged(self)

    def optimisation_nearly_converged(self) -> bool:
        """
        Check whether a calculation has nearly converged and may just need
        more geometry optimisation steps to complete successfully

        Returns:
            (bool):
        """
        logger.info('Checking to see if the geometry nearly converged')
        if not self.output.exists:
            return False

        return self.method.optimisation_nearly_converged(self)

    def get_final_atoms(self) -> Atoms:
        """
        Get the atoms from the final step of a geometry optimisation

        Returns:
            (autode.atoms.Atoms):

        Raises:
            (autode.exceptions.AtomsNotFound):
        """
        logger.info(f'Getting final atoms from {self.output.filename}')

        if not self.output.exists:
            logger.error('No calculation output. Could not get atoms')
            raise ex.AtomsNotFound

        atoms = Atoms(self.method.get_final_atoms(self))

        if len(atoms) != self.molecule.n_atoms:
            logger.error(f'Failed to get atoms from {self.output.filename}')
            raise ex.AtomsNotFound

        return atoms

    def get_atomic_charges(self) -> List[float]:
        """
        Get the partial atomic charges from a calculation. The method used to
        calculate them depends on the QM method and are implemented in their
        respective wrappers

        Returns:
            (list(float)): Atomic charges in units of e
        """
        if not self.output.exists:
            logger.error('No calculation output. Could not get final charges')
            raise ex.CouldNotGetProperty(name='atomic charges')

        logger.info(f'Getting atomic charges from {self.output.filename}')
        charges = self.method.get_atomic_charges(self)

        if len(charges) != self.molecule.n_atoms:
            raise ex.CouldNotGetProperty(name='atomic charges')

        return charges

    def get_gradients(self) -> Gradient:
        """
        Get the gradient (dE/dr) with respect to atomic displacement from a
        calculation

        Returns:
            (autode.values.Gradient): Gradient vectors. shape = (n_atoms, 3)

        Raises:
            (autode.exceptions.CouldNotGetProperty):
        """
        logger.info(f'Getting gradients from {self.output.filename}')
        gradients = Gradient(self.method.get_gradients(self))

        if len(gradients) != self.molecule.n_atoms:
            raise ex.CouldNotGetProperty(name='gradients')

        return gradients

    def get_hessian(self) -> Hessian:
        """
        Get the Hessian matrix (d^2E/dr^2) i.e. the matrix of second
        derivatives of the energy with respect to cartesian displacements::

            H =  (d^2E/dx_0^2, d^2E/dx_0dy_0, d^2E/dx_0dz_0, d^2E/dx_0dx_1 ...
                  d^2E/dy_0dx_0      .               .              .
                      .              .               .              . )

        Returns:
            (autode.values.Hessian): Hessian matrix. shape = (3N, 3N) for N atoms

        Raises:
            (autode.exceptions.CouldNotGetProperty):
        """
        logger.info(f'Getting Hessian from calculation')

        try:
            hessian = Hessian(self.method.get_hessian(self),
                              atoms=self.get_final_atoms(),
                              units='Ha/ang^2')

            assert hessian.shape == (3*self.molecule.n_atoms,
                                     3*self.molecule.n_atoms)

        except (ValueError, IndexError, AssertionError) as err:
            raise ex.CouldNotGetProperty(f'Could not get the Hessian: {err}')

        return hessian

    @property
    def terminated_normally(self) -> bool:
        """
        Determine if the calculation terminated without error

        Returns:
            (bool): Normal termination of the calculation?
        """
        logger.info(f'Checking for {self.output.filename} normal termination')

        if not self.output.exists:
            logger.warning('Calculation did not generate any output')
            return False

        return self.method.calculation_terminated_normally(self)

    def clean_up(self,
                 force:      bool = False,
                 everything: bool = False) -> None:
        """
        Clean up input and output files, if Config.keep_input_files is False
        (and not force=True)

        Keyword Arguments:

            force (bool): If True then override Config.keep_input_files

            everything (bool): Remove both input and output files
        """

        if Config.keep_input_files and not force:
            logger.info('Keeping input files')
            return

        filenames = self.input.filenames
        if everything:
            filenames.append(self.output.filename)

        logger.info(f'Deleting {filenames}')

        # Delete the files that exist
        for filename in filenames:
            if not os.path.exists(filename):
                logger.warning(f'Could not delete {filename} it did not exist')
                continue
            os.remove(filename)

        return None

    def generate_input(self) -> None:
        """Generate the required input"""
        logger.info(f'Generating input file(s) for {self.name}')

        # Can switch off uniqueness testing with e.g.
        # export AUTODE_FIXUNIQUE=False   used for testing
        if os.getenv('AUTODE_FIXUNIQUE', True) != 'False':
            self._fix_unique()

        self.input.filename = self.method.get_input_filename(self)

        # Check that if the keyword is a autode.wrappers.keywords.Keyword then
        # it has the required name in the method used for this calculation
        for keyword in self.input.keywords:
            # Allow keywords as strings
            if not isinstance(keyword, kws.Keyword):
                continue

            # Allow for the unambiguous setting of a keyword with only a name
            if keyword.has_only_name():
                # set e.g. keyword.orca = 'b3lyp'
                setattr(keyword, self.method.name, keyword.name)
                continue

            # For a keyword e.g. Keyword(name='pbe', orca='PBE') then the
            # definition in this method is not obvious, so raise an exception
            if not hasattr(keyword, self.method.name):
                err_str = (f'Keyword: {keyword} is not supported set '
                           f'{keyword}.{self.method.name} as a string')
                raise ex.UnsuppportedCalculationInput(err_str)

        self.method.generate_input(self, self.molecule)

        return None

    def execute_calculation(self) -> None:
        """
        Execute a calculation if it has not been run, or if it did not finish
        with a normal termination
        """
        logger.info(f'Running {self.input.filename} using {self.method.name}')

        if not self.input.exists:
            raise ex.NoInputError('Input did not exist')

        if self.output.exists and self.terminated_normally:
            logger.info('Calculation already terminated normally. Skipping')
            return None

        # Check that the method used to execute the calculation is available
        if not self.method.available:
            raise ex.MethodUnavailable

        self.method.execute(self)

        return None

    def run(self) -> None:
        """Run the calculation using the EST method """
        logger.info(f'Running calculation {self.name}')

        self.generate_input()
        self.output.filename = self.method.get_output_filename(self)
        self.execute_calculation()
        self.clean_up()
        self._add_to_comp_methods()

        return None

    def print_final_output_lines(self, n: int = 50) -> None:
        """
        Print the final n output lines, if the output exists

        -----------------------------------------------------------------------
        Arguments:
            n: Number of lines
        """

        if self.output.exists:
            print("\n".join(self.output.file_lines[-n:]))

        return None


class CalculationOutput:

    @cached_property
    def file_lines(self):
        """
        Output files lines. This may be slow for large files but should
        not become a bottleneck when running standard DFT/WF calculations,
        are cached so only read once

        Returns:
            (list(str)): Lines from the output file

        Raises:
            (autode.exceptions.NoCalculationOutput):
        """
        logger.info('Setting output file lines')

        if self.filename is None or not os.path.exists(self.filename):
            raise ex.NoCalculationOutput

        return open(self.filename, 'r', encoding="utf-8").readlines()

    @property
    def exists(self):
        """Does the calculation output exist?"""
        return self.filename is not None and os.path.exists(self.filename)

    def __init__(self, filename: Optional[str] = None):

        self.filename = filename


class CalculationInput:

    def __init__(self,
                 keywords:        'autode.wrappers.keywords.Keywords',
                 additional_input: Optional[str] = None,
                 added_internals:  Optional[list] = None,
                 point_charges:    Optional[List[PointCharge]] = None):
        """
        Arguments:
            keywords (autode.wrappers.keywords.Keywords):

            additional_input (str or None): Any additional input string to add
                                            to the input file, or None

            added_internals (list(tuple(int)) or None): Atom indexes to add to
                                                       the internal coordinates

            point_charges (list(autode.point_charges.PointCharge) or None):
                          list of float of point charges, x, y, z coordinates
                          for each point charge
        """
        self.keywords = keywords
        self.other_block = additional_input

        self.added_internals = added_internals
        self.point_charges = point_charges

        self.filename = None
        self.additional_filenames = []

        self._check()

    def _check(self):
        """Check that the input parameters have the expected format"""
        if self.keywords is not None:
            assert isinstance(self.keywords, kws.Keywords)

        assert self.other_block is None or type(self.other_block) is str

        # Ensure the point charges are given as a list of PointCharge objects
        if self.point_charges is not None:
            assert type(self.point_charges) is list
            assert all(type(pc) is PointCharge for pc in self.point_charges)

        if self.added_internals is not None:
            assert type(self.added_internals) is list
            assert all(len(idxs) == 2 for idxs in self.added_internals)

    @property
    def exists(self):
        """Does the input (files) exist?"""
        return (self.filename is not None
                and all(os.path.exists(fn) for fn in self.filenames))

    @property
    def filenames(self):
        """Return a list of all the input files"""
        if self.filename is None:
            return self.additional_filenames

        return [self.filename] + self.additional_filenames
