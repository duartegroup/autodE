from copy import deepcopy
import os
from autode.point_charges import PointCharge
from autode.solvent.solvents import get_available_solvent_names
from autode.config import Config
from autode.exceptions import AtomsNotFound
from autode.exceptions import CouldNotGetProperty
from autode.exceptions import MethodUnavailable
from autode.exceptions import NoInputError
from autode.exceptions import SolventUnavailable
from autode.exceptions import NoCalculationOutput
from autode.log import logger

output_exts = ('.out', '.hess', '.xyz', '.inp', '.com', '.log', '.nw',
               '.pc', '.grad')


# Top level function that can be hashed
def execute_calc(calc):
    return calc.execute_calculation()


def get_solvent_name(molecule, method):
    """
    Set the solvent keyword to use in the calculation given an QM method

    Arguments:
        molecule (autode.species.Species)
        method (autode.wrappers.base.ElectronicStructureMethod):
    """
    available_solvents = (f'Available solvents for {method.__name__} are '
                          f'{get_available_solvent_names(method)}')

    if molecule.solvent is None:
        logger.info('Calculation is in the gas phase')
        return None

    solvent_name = getattr(molecule.solvent, method.name)
    if solvent_name is None:
        raise SolventUnavailable(message=available_solvents)

    return solvent_name


class Calculation:

    def _check(self):
        """Ensure the calculation and molecule has the required attributes"""
        assert hasattr(self.molecule, 'n_atoms')
        assert hasattr(self.molecule, 'atoms')
        assert hasattr(self.molecule, 'mult')
        assert hasattr(self.molecule, 'charge')
        assert hasattr(self.molecule, 'solvent')

        # The molecule must have > 0 atoms
        if self.molecule.atoms is None or self.molecule.n_atoms == 0:
            logger.error('Have no atoms. Can\'t form a calculation')
            raise NoInputError

    def get_energy(self, e=True, h=False, g=False, force=False):
        """
        Get the energy from a completed calculation

        Keyword Arguments:
            e (bool): Return the potential energy (E)
            h (bool): Return the enthalpy (H) at 298 K
            g (bool): Return the Gibbs free energy (G) at 298 K
            force (bool): Return the energy even if the calculation errored

        Returns:
            (float): Energy in Hartrees, or None
        """

        logger.info(f'Getting energy from {self.output.filename}')
        if self.terminated_normally or force:

            if e:
                return self.method.get_energy(self)

            if h:
                return self.method.get_enthalpy(self)

            if g:
                return self.method.get_free_energy(self)

        logger.error('Calculation did not terminate normally. Energy = None')
        return None

    def optimisation_converged(self):
        """Check whether a calculation has has converged to within the theshold
        on energies and graidents specified in the input

        Returns:
            (bool)
        """

        logger.info('Checking to see if the geometry converged')
        return self.method.optimisation_converged(self)

    def optimisation_nearly_converged(self):
        """Check whether a calculation has nearly converged and may just need
        more geometry optimisation steps to complete successfully

        Returns:
            (bool)
        """
        logger.info('Checking to see if the geometry nearly converged')
        return self.method.optimisation_nearly_converged(self)

    def get_imaginary_freqs(self):
        """Get the imaginary frequencies from a calculation output note that
        they are returned as negative to conform with standard QM codes

        Returns:
            (list(float)): List of negative frequencies in wavenumbers (cm-1)
        """
        logger.info(f'Getting imaginary frequencies from {self.name}')
        return self.method.get_imaginary_freqs(self)

    def get_normal_mode_displacements(self, mode_number):
        """Get the displacements along a mode for each of the n_atoms in the
        structure will return a list of length n_atoms each with 3 components
        (x, y, z)

        Arguments:
            mode_number (int): Normal mode number. 6 will be the first
                               vibrational mode as 0->2 are translation and
                               3->5 rotation
        Returns:
            (np.ndarray): Displacement vectors for each atom (Å)
                          modes.shape = (n_atoms, 3)
        """
        modes = self.method.get_normal_mode_displacements(self, mode_number)

        if len(modes) != self.molecule.n_atoms:
            raise CouldNotGetProperty(name='normal modes')

        return modes

    def get_final_atoms(self):
        """
        Get the atoms from the final step of a geometry optimisation or the
        first (only) step of a single point calculation

        Returns:
            (list(autode.atoms.Atom)):
        """
        logger.info(f'Getting final atoms from {self.output.filename}')

        if not self.output.exists():
            logger.error('No calculation output. Could not get atoms')
            raise AtomsNotFound

        # Extract the atoms from the output file, which is method dependent
        atoms = self.method.get_final_atoms(self)

        if len(atoms) != self.molecule.n_atoms:
            logger.error(f'Failed to get atoms from {self.output.filename}')
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self):
        """
        Get the partial atomic charges from a calculation. The method used to
        calculate them depends on the QM method and are implemented in their
        respective wrappers

        Returns:
            (list(float)): Atomic charges in units of e
        """
        logger.info(f'Getting atomic charges from {self.output.filename}')
        charges = self.method.get_atomic_charges(self)

        if len(charges) != self.molecule.n_atoms:
            raise CouldNotGetProperty(name='atomic charges')

        return charges

    def get_gradients(self):
        """
        Get the gradient (dE/dr) with respect to atomic displacement from a
        calculation

        Returns:
            (np.ndarray): Gradient vectors for each atom (Å)
                          gradients.shape = (n_atoms, 3)
        """
        logger.info(f'Getting gradients from {self.output.filename}')
        gradients = self.method.get_gradients(self)

        if len(gradients) != self.molecule.n_atoms:
            raise CouldNotGetProperty(name='gradients')

        return gradients

    def terminated_normally(self):
        """Determine if the calculation terminated without error"""
        logger.info(f'Checking for {self.output.filename} normal termination')

        if not self.output.exists():
            logger.warning('Calculation did not generate any output')
            return False

        if self.method.terminated_normally(self):
            return True
        else:
            return False

    def clean_up(self):
        """Clean up input files, if Config.keep_input_files is False"""

        if Config.keep_input_files:
            return None
        else:
            return self.method.clean_up(self)

    def generate_input(self):
        """Generate the required input and set the output filename"""
        self.method.generate_input(self.input, self.molecule, self.n_cores)
        return None

    def execute_calculation(self):
        """Execute a calculation if it has not been run or finish correctly"""
        logger.info(f'Running {self.input.filename} using {self.method.name}')

        if not self.input.exists():
            raise NoInputError

        # Check that the method used to execute the calculation is available
        self.method.set_availability()
        if not self.method.available:
            raise MethodUnavailable

        # If the output file already exists set the output lines
        if os.path.exists(self.output.filename):
            self.output.set_lines()

        if self.output.exists() and self.terminated_normally():
            logger.info('Calculation already terminated normally. Skipping')
            return None


        """
        logger.info(f'Setting the number of OMP threads to {self.n_cores}')
        os.environ['OMP_NUM_THREADS'] = str(self.n_cores)
        
        @work_in_tmp_dir(filenames_to_copy=[self.input_filename]+self.additional_input_files, kept_file_exts=output_exts)
        def execute_est_method():

            with open(self.output_filename, 'w') as output_file:

                if self.method.mpirun:
                    params = ['mpirun', '-np', str(self.n_cores), self.method.path, self.input_filename]
                else:
                    params = [self.method.path, self.input_filename]
                if self.flags is not None:
                    params += self.flags

                subprocess = Popen(params, stdout=output_file, stderr=open(os.devnull, 'w'))
            subprocess.wait()
            logger.info(f'Calculation {self.output_filename} done')
            if self.grad and self.method.name == 'xtb':
                # Need to get the XTB gradients
                self.get_gradients()
        """

        self.method.execute(self)
        self.output.set_lines()

        return None

    def run(self):
        """Run the calculation using the EST method """
        logger.info(f'Running calculation {self.name}')

        # Set an input filename and generate the input
        self.input.filename = self.method.get_input_filename(self)
        self.generate_input()

        # Set the output filename, run the calculation and clean up the files
        self.output.filename = self.method.get_output_filename(self)
        self.execute_calculation()
        self.clean_up()

        return None

    def __init__(self, name, molecule, method, keywords=None, n_cores=1,
                 bond_ids_to_add=None,
                 other_input_block=None,
                 distance_constraints=None,
                 cartesian_constraints=None,
                 point_charges=None):
        """
        Arguments:
            name (str): calc name
            molecule (molecule object): molecule to be calculated
            method (method object): which electronic structure wrapper to use

        Keyword Arguments:
            keywords (autode.wrappers.keywords.Keywords): Keywords to use
                                                          (default: {None})

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
        self.name = name

        # ------------------- System specific parameters ----------------------
        self.molecule = deepcopy(molecule)

        self.molecule.constraints = Constraints(distance=distance_constraints,
                                                cartesian=cartesian_constraints)

        # --------------------- Calculation parameters ------------------------
        self.method = method
        self.n_cores = int(n_cores)

        # ------------------- Calculation input/output ------------------------
        self.input = CalculationInput(keywords=keywords,
                                      solvent=get_solvent_name(molecule, method),
                                      additional_input_block=other_input_block,
                                      added_internals=bond_ids_to_add,
                                      point_charges=point_charges)

        self.output = CalculationOutput()

        # Check attribute types and self.molecule
        self._check()


class CalculationOutput:

    def set_lines(self):
        """
        Set the output files lines. This may be slow for large files but should
        not become a bottleneck when running standard DFT/WF calculations

        Returns:
            (None)
        """

        if not os.path.exists(self.filename):
            raise NoCalculationOutput

        self.file_lines = open(self.filename, 'r', encoding="utf-8").readlines()
        self.rev_file_lines = reversed(self.file_lines)

        return None

    def exists(self):
        """Does the calculation output exist?"""

        if self.filename is None or self.file_lines is None:
            return False

        return True

    def __init__(self):

        self.filename = None
        self.file_lines = None
        self.rev_file_lines = None


class CalculationInput:

    def _check(self):
        """Check that the input parameters have the expected format"""
        assert self.keywords is not None
        assert self.solvent is None or type(self.solvent) is str
        assert self.other_block is None or type(self.other_block) is str

        # Ensure the point charges are given as a list of PointCharge objects
        if self.point_charges is not None:
            assert type(self.point_charges) is list
            assert all(type(pc) is PointCharge for pc in self.point_charges)

        if self.added_internals is not None:
            assert type(self.added_internals) is list
            assert all(len(idxs) == 2 for idxs in self.added_internals)

    def exists(self):
        """Does the input (files) exist?"""

        if self.filename is None:
            return False

        return all(os.path.exists(fn) for fn in self.get_input_filenames())

    def get_input_filenames(self):
        """Return a list of all the input files"""
        assert self.filename is not None
        return [self.filename] + self.additional_filenames

    def __init__(self, keywords, solvent, additional_input_block,
                 added_internals, point_charges):
        """
        Args:
            keywords (autode.wrappers.keywords.Keywords):

            solvent (str): Name of the solvent for this QM method, or None

            additional_input_block (str): Any additional input string to add
                                          to the input file, or None

            added_internals (list(tuple(int))): Atom indexes to add to the
                                                internal coordinates, or None

            point_charges (list(autode.point_charges.PointCharge)): list of
                                 float of point charges, x, y, z
                                 coordinates for each point charge
        """
        self.keywords = keywords
        self.solvent = solvent
        self.other_block = additional_input_block

        self.added_internals = added_internals
        self.point_charges = point_charges

        self.filename = None
        self.additional_filenames = []

        self._check()


class Constraints:

    def _check(self):
        """ Check the constraints have the expected format"""
        assert type(self.distance) is dict
        assert all(len(key) == 2 for key in self.distance.keys())

        assert type(self.cartesian) is list
        assert all(type(item) is int for item in self.cartesian)

    def __init__(self, distance, cartesian):
        """
        Args:
            distance (dict): Keys of: tuple(int) for two atom indexes and
                             values of the distance in Å or None
            cartesian (list(int)): List of atom indexes or None
        """

        self.distance = distance if distance is not None else {}
        self.cartesian = cartesian if cartesian is not None else []

        self._check()
