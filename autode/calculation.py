import os
from subprocess import Popen
from autode.log import logger
from autode.exceptions import AtomsNotFound
from autode.exceptions import CouldNotGetProperty
from autode.exceptions import NoInputError
from autode.exceptions import MethodUnavailable
from autode.utils import work_in_tmp_dir
from autode import mol_graphs
from autode.config import Config
from autode.solvent.solvents import get_available_solvent_names
from autode.exceptions import SolventUnavailable
from copy import deepcopy

output_exts = ('.out', '.hess', '.xyz', '.inp', '.com', '.log', '.nw', '.pc', '.grad')


# Top level function that can be hashed
def execute_calc(calc):
    return calc.execute_calculation()


def check_molecule_attr(molecule):
    assert hasattr(molecule, 'n_atoms')
    assert hasattr(molecule, 'atoms')
    assert hasattr(molecule, 'mult')
    assert hasattr(molecule, 'charge')
    assert hasattr(molecule, 'solvent')


class Calculation:

    def _set_core_atoms(self, molecule):
        """Finds the atoms involved in the reaction, and those bonded to them. These atoms are then
        calculated exactly in the hybrid hessian, if a full exact hessian is not calculated

        Arguments:
            molecule (mol obj): the molecule being calculated
        """
        active_atoms = set()
        for bond in self.bond_ids_to_add:
            active_atoms.add(bond[0])
            active_atoms.add(bond[1])

        # Will set molecule.graph to a NetworkX Graph to find the nearest neighbours
        if hasattr(molecule, 'graph') and molecule.graph is not None:
            logger.warning('Molecular graph was set. Not regenerating')
        else:
            mol_graphs.make_graph(molecule)

        core_atoms = set()
        for active_atom in active_atoms:
            bonded_atoms = molecule.graph.neighbors(active_atom)
            core_atoms.add(active_atom)

            for bonded_atom in bonded_atoms:
                core_atoms.add(bonded_atom)

        self.core_atoms = list(core_atoms)
        return None

    def get_energy(self):
        logger.info(f'Getting energy from {self.output_filename}')
        if self.terminated_normally:
            return self.method.get_energy(self)

        else:
            logger.error('Calculation did not terminate normally – not returning the energy')
            return None

    def optimisation_converged(self):
        logger.info('Checking to see if the geometry converged')
        return self.method.optimisation_converged(self)

    def optimisation_nearly_converged(self):
        """Check whether a calculation has nearly converged and may just need more geometry optimisation steps to
        complete successfully

        Returns:
            bool: if the calc is nearly converged or not
        """
        return self.method.optimisation_nearly_converged(self)

    def get_imag_freqs(self):
        logger.info('Finding imaginary frequencies in cm-1')
        return self.method.get_imag_freqs(self)

    def get_normal_mode_displacements(self, mode_number):
        """Get the displacements along a mode for each of the n_atoms in the structure will return a list of length
        n_atoms each with 3 components (x, y, z)

        Arguments:
            mode_number (int): normal mode number. 6 will be the first vibrational mode (indexed from 0 in orca)

        Returns:
            list(list): list of displacement distances for each xyz
        """
        return self.method.get_normal_mode_displacements(self, mode_number)

    def get_final_atoms(self):
        logger.info(f'Getting final xyzs from {self.output_filename}')

        if self.output_file_lines is None or self.output_filename is None:
            logger.error('Could not get the final atoms. The output file lines were not set')
            raise AtomsNotFound

        atoms = self.method.get_final_atoms(self)

        if len(atoms) == 0:
            logger.error(f'Could not get xyzs from calculation file {self.name}')
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self):
        logger.info(f'Getting atomic charges from calculation file {self.output_filename}')
        charges = self.method.get_atomic_charges(self)

        if len(charges) != self.molecule.n_atoms:
            raise  CouldNotGetProperty(f'Could not get atomic charges from calculation output file {self.name}')

        return charges

    def get_gradients(self):
        logger.info(f'Getting gradients from calculation file {self.output_filename}')
        gradients = self.method.get_gradients(self)

        if len(gradients) != self.molecule.n_atoms:
            raise CouldNotGetProperty(f'Could not get gradients from calculation output file {self.name}')

        return gradients

    def calculation_terminated_normally(self):
        logger.info(f'Checking to see if {self.output_filename} terminated normally')

        if self.output_file_lines is None:
            logger.warning('Calculation did not generate any output')
            return False

        return self.method.calculation_terminated_normally(self)

    def set_output_file_lines(self):

        self.output_file_lines = [line for line in open(self.output_filename, 'r', encoding="utf-8")]
        self.rev_output_file_lines = list(reversed(self.output_file_lines))

        return None

    def generate_input(self):
        logger.info(f'Generating input file for {self.name}')
        return self.method.generate_input(self)

    def execute_calculation(self):
        logger.info(f'Running calculation {self.input_filename} using {self.method.name}')

        if self.input_filename is None:
            raise NoInputError

        if not os.path.exists(self.input_filename):
            raise NoInputError

        self.method.set_availability()
        if not self.method.available:
            raise MethodUnavailable

        if os.path.exists(self.output_filename):
            self.output_file_exists = True
            self.set_output_file_lines()

        if self.output_file_exists:
            if self.calculation_terminated_normally():
                logger.info('Calculation already terminated successfully. Skipping')
                return

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

        execute_est_method()
        return self.set_output_file_lines()

    def run(self):
        logger.info(f'Running calculation of {self.name}')

        self.generate_input()
        self.execute_calculation()
        self.terminated_normally = self.calculation_terminated_normally()

        return None

    def __init__(self, name, molecule, method, keywords_list=None, n_cores=1, bond_ids_to_add=None,
                 other_input_block=None, opt=False, distance_constraints=None, cartesian_constraints=None,
                 constraints_already_met=False, grad=False):
        """
        Arguments:
            name (str): calc name
            molecule (molecule object): molecule to be calculated
            method (method object): which electronic structure wrapper to use

        Keyword Arguments:
            keywords_list (list(str)): keywords_list to use in the calc (default: {None})
            n_cores (int): number of cores available (default: {1})
            bond_ids_to_add (list(tuples)): list of active bonds (default: {None})
            other_input_block (list): keywords_list to use when performing a TS search (default: {None})
            opt (bool): opt calc or not (needed for xtb) (default: {False})
            distance_constraints (dict): keys = tuple of atom ids for a bond to be kept at fixed length, value = length
                                         to be fixed at (default: {None})
            cartesian_constraints (list(int)): list of atom ids to fix at their cartesian coordinates (default: {None})
            constraints_already_met (bool): if the constraints are already met, or need optimising to (needed for xtb force constant) (default: {False})
            grad (bool): grad calc or not (needed for xtb) (default: {False})
        """

        # TODO Purge some of these attributes
        self.name = name
        check_molecule_attr(molecule=molecule)
        self.molecule = deepcopy(molecule)

        self.method = method
        self.keywords_list = keywords_list
        self.flags = None
        self.opt = opt
        self.core_atoms = None
        self.grad = grad

        self.n_cores = n_cores
        self.max_core_mb = Config.max_core        # Maximum memory per core to use

        self.bond_ids_to_add = bond_ids_to_add
        self.other_input_block = other_input_block
        self.distance_constraints = distance_constraints
        self.cartesian_constraints = cartesian_constraints
        self.constraints_already_met = constraints_already_met

        # Set in self.generate_input()
        self.input_filename = None
        # Set in self.generate_input()
        self.output_filename = None

        self.output_file_exists = False
        self.terminated_normally = False
        self.output_file_lines = None
        self.rev_output_file_lines = None

        self.additional_input_files = []

        self.solvent_keyword = None
        if molecule.solvent is not None:
            if getattr(molecule.solvent, method.name) is None:
                print(f'Available solvents for {self.method.__name__} are {get_available_solvent_names(self.method)}')
                raise SolventUnavailable

            self.solvent_keyword = getattr(molecule.solvent, method.name)

        if self.molecule.atoms is None or self.molecule.n_atoms == 0:
            logger.error('Have no xyzs. Can\'t make a calculation')
            raise NoInputError

        if self.bond_ids_to_add:
            self._set_core_atoms(molecule)
