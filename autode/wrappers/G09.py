from copy import deepcopy
import numpy as np
import autode.wrappers.keywords as kws
from autode.constants import Constants
from autode.wrappers.base import ElectronicStructureMethod
from autode.utils import run_external
from autode.atoms import Atom
from autode.config import Config
from autode.exceptions import AtomsNotFound
from autode.log import logger
from autode.calculation import CalculationOutput
from autode.calculation import Constraints
from autode.utils import work_in_tmp_dir


def modify_keywords_for_point_charges(keywords):
    """For a list of Gaussian keywords modify to include z-matrix if not
    already included. Required if point charges are included in the calc"""
    logger.warning('Modifying keywords as point charges are present')

    keywords.append('Charge')

    for keyword in keywords:
        if 'opt' not in keyword.lower():
            continue

        opt_options = []
        if '=(' in keyword:
            # get the individual options
            unformated_options = keyword[5:-1].split(',')
            opt_options = [option.lower().strip() for option in unformated_options]

        elif '=' in keyword:
            opt_options = [keyword[4:]]

        if not any(option.lower() == 'z-matrix' for option in opt_options):
            opt_options.append('Z-Matrix')

        new_keyword = f'Opt=({", ".join(opt_options)})'
        keywords.remove(keyword)
        keywords.append(new_keyword)

    return None


def get_keywords(calc_input, molecule):
    """Modify the input keywords to try and fix some Gaussian's quirks"""

    new_keywords = []   # List of keywords as strings for this calculation

    for keyword in calc_input.keywords.copy():

        # Add any empirical dispersion
        if isinstance(keyword, kws.DispersionCorrection):
            new_keywords.append(f'EmpiricalDispersion={keyword.g09}')

        # Replace the basis set file specification with genecp
        elif str(keyword).endswith('.gbs'):
            logger.info('Found a custom basis set file adding genecp')
            new_keywords.append('genecp')

        # and any other keywords, that may be a Keyword with a g09
        # attribute or just a name, or just a string
        elif isinstance(keyword, kws.Keyword):
            new_keywords.append(keyword.g09)

        else:
            new_keywords.append(str(keyword))

    # Mod redundant keywords is required if there are any constraints or
    # modified internal coordinates
    if molecule.constraints.any():
        new_keywords.append('Geom=ModRedun')

    if calc_input.added_internals is not None:
        new_keywords.append('Geom=ModRedun')

    # Remove the optimisation keyword if there is only a single atom
    opt = False
    for keyword in new_keywords:

        if 'opt' not in keyword.lower():
            continue

        opt = True

        if molecule.n_atoms == 1:
            logger.warning('Cannot do an optimisation for a single atom')
            new_keywords.remove(keyword)

    if calc_input.temp is not None:
        new_keywords.append(f'Temperature={calc_input.temp:.2f}')

    # Further modification is required if there are surrounding point charges
    if calc_input.point_charges is not None:
        modify_keywords_for_point_charges(new_keywords)

    # By default perform all optimisations without symmetry
    if opt and not any(kw.lower() == 'nosymm' for kw in new_keywords):
        if hasattr(molecule, 'is_linear') and molecule.is_linear():
            # Allow symmetry for linear molecules so the free energy
            # calculation
            pass
        else:
            new_keywords.append('NoSymm')

    return new_keywords


def print_point_charges(inp_file, calc_input):
    """Add point charges to the input file"""

    if calc_input.point_charges is None:
        return

    print('', file=inp_file)
    for point_charge in calc_input.point_charges:
        x, y, z = point_charge.coord
        print(f'{x:^12.8f} {y:^12.8f} {z:^12.8f} {point_charge.charge:^12.8f}',
              file=inp_file)
    return


def print_added_internals(inp_file, calc_input):
    """Add any internal coordinates to the input file"""

    if calc_input.added_internals is None:
        return

    for (i, j) in calc_input.added_internals:
        # Gaussian indexes atoms from 1
        print('B', i + 1, j + 1, file=inp_file)

    return


def print_constraints(inp_file, molecule):
    """Add any distance or cartesian constraints to the input file"""

    if molecule.constraints.distance is not None:

        for (i, j), dist in molecule.constraints.distance.items():
            # Gaussian indexes atoms from 1
            print('B', i + 1, j + 1, dist, 'B', file=inp_file)
            print('B', i + 1, j + 1, 'F', file=inp_file)

    if molecule.constraints.cartesian is not None:

        for i in molecule.constraints.cartesian:
            # Gaussian indexes atoms from 1
            print('X', i+1, 'F', file=inp_file)
    return


def print_custom_basis(inp_file, keywords):
    """Print the definition of the custom basis set file """

    for keyword in keywords:
        if isinstance(keyword, kws.Keyword) and hasattr(keyword, 'g09'):
            str_keyword = keyword.g09
        else:
            str_keyword = str(keyword)

        if str_keyword.endswith('.gbs'):
            print(f'@{keyword}', file=inp_file)

    return None


def rerun_angle_failure(calc):
    """
    Gaussian will sometimes encounter a 180 degree angle and crash. This
    function performs a few geometry optimisation cycles in cartesian
    coordinates then switches back to internals

    Arguments:
        calc (autode.calculation.Calculation):

    Returns:
        (autode.calculation.Calculation):
    """
    cart_calc = deepcopy(calc)

    # Iterate through a copied set of keywords
    for keyword in cart_calc.input.keywords.copy():
        if keyword.lower().startswith('geom'):
            cart_calc.input.keywords.remove(keyword)

        elif keyword.lower().startswith('opt'):
            options = []
            if '=(' in keyword:
                # get the individual options
                options = [option.lower().strip()
                           for option in keyword[5:-1].split(',')]

                for option in options:
                    if (option.startswith('maxcycles')
                            or option.startswith('maxstep')):
                        options.remove(option)

            elif '=' in keyword:
                options = [keyword[4:]]
            options += ['maxcycles=3', 'maxstep=1', 'cartesian']

            new_keyword = f'Opt=({", ".join(options)})'
            cart_calc.input.keywords.remove(keyword)
            cart_calc.input.keywords.append(new_keyword)

    # Generate the new calculation and run
    cart_calc.name += '_cartesian'
    cart_calc.molecule.atoms = calc.get_final_atoms()
    cart_calc.molecule.constraints = Constraints(distance=None, cartesian=None)
    cart_calc.input.added_internals = None
    cart_calc.output = CalculationOutput()
    cart_calc.run()

    if not cart_calc.terminated_normally():
        logger.warning('Cartesian calculation did not converge')
        return None

    logger.info('Returning to internal coordinates')

    # Reset the required parameters for the new calculation
    fixed_calc = deepcopy(calc)
    fixed_calc.name += '_internal'
    fixed_calc.molecule.atoms = cart_calc.get_final_atoms()
    fixed_calc.output = CalculationOutput()
    fixed_calc.run()

    return fixed_calc


class G09(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):
        """Print a Gaussian input file"""

        with open(calc.input.filename, 'w') as inp_file:

            # Gaussian defines the total memory for the whole calculation, not
            # per core
            total_mem = int(Config.max_core * calc.n_cores)
            print(f'%mem={total_mem}MB', file=inp_file)

            if calc.n_cores > 1:
                print(f'%nprocshared={calc.n_cores}', file=inp_file)

            keywords = get_keywords(calc.input, molecule)
            print('#', *keywords, file=inp_file, end=' ')

            if calc.input.solvent is not None:
                print(f'scrf=(smd,solvent={calc.input.solvent})', file=inp_file)
            else:
                print('', file=inp_file)

            print(f'\n {calc.name}\n', file=inp_file)

            print(molecule.charge, molecule.mult, file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
                      file=inp_file)

            print_point_charges(inp_file, calc.input)
            print('', file=inp_file)
            print_added_internals(inp_file, calc.input)
            print_constraints(inp_file, molecule)
            print_custom_basis(inp_file, calc.input.keywords)
            # Gaussian needs blank lines at the end of the file
            print('\n', file=inp_file)

        return None

    def get_input_filename(self, calc):
        return f'{calc.name}.com'

    def get_output_filename(self, calc):
        return f'{calc.name}.log'

    def get_version(self, calc):
        """Get the version of Gaussian used in this calculation"""

        for line in calc.output.file_lines:

            if line.startswith('Gaussian ') and 'Revision' in line:
                return line.lstrip('Gaussian ')

        logger.warning('Could not find the Gaussian version number')
        return '???'

    def execute(self, calc):

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.log', '.com'))
        def execute_g09():
            run_external(params=[calc.method.path, calc.input.filename],
                         output_filename=calc.output.filename)

        execute_g09()
        return None

    def calculation_terminated_normally(self, calc, rerun_if_failed=True):

        termination_strings = ['Normal termination of Gaussian',
                               'Number of steps exceeded']

        bend_ok = True  # Gaussian can fail when 180º bends are encountered
        for line in reversed(calc.output.file_lines):

            if any(string in line for string in termination_strings):
                logger.info('Gaussian terminated normally')
                return True

            if 'Bend failed for angle' in line:
                logger.warning('Gaussian encountered a 180° angle and crashed')
                bend_ok = False
                break

        if bend_ok or not rerun_if_failed:
            return False

        # Set a limit on the amount of times we do this
        if calc.name.endswith('internal_internal_internal_internal'):
            return False

        try:
            # To fix the calculation requires the atoms to be in the output
            fixed_calc = rerun_angle_failure(calc)

        except AtomsNotFound:
            return False

        if fixed_calc is not None and fixed_calc.terminated_normally():
            logger.info('The 180° angle issue has been fixed')
            calc.output = fixed_calc.output
            calc.name = fixed_calc.name
            calc.output.set_lines()
            return True

        return False

    def get_enthalpy(self, calc):
        """Get the enthalpy (H) from an g09 calculation output"""

        for line in reversed(calc.output.file_lines):
            if 'Sum of electronic and thermal Enthalpies' in line:
                return float(line.split()[-1])

        logger.error('Could not get the enthalpy from the calculation. '
                     'A frequency must be requested')
        return None

    def get_free_energy(self, calc):
        """Get the Gibbs free energy (G) from an g09 calculation output"""

        for line in reversed(calc.output.file_lines):
            if 'Sum of electronic and thermal Free Energies' in line:
                return float(line.split()[-1])

        logger.error('Could not get the enthalpy from the calculation. '
                     'A frequency must be requested')
        return None

    def get_energy(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'SCF Done' in line:
                return float(line.split()[4])
            if 'E(CORR)' in line:
                return float(line.split()[3])
            if 'E(CI)' in line:
                return float(line.split()[3])
            if 'E(CIS)' in line:
                return float(line.split()[4])
            if 'E(CIS(D))' in line:
                return float(line.split()[5])

        return None

    def optimisation_converged(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'Optimization completed' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        geom_conv_block = False

        for line in reversed(calc.output.file_lines):
            if geom_conv_block and 'Item' in line:
                geom_conv_block = False
            if 'Predicted change in Energy' in line:
                geom_conv_block = True
            if geom_conv_block and len(line.split()) == 4:
                if line.split()[-1] == 'YES':
                    return True
        return False

    def get_imaginary_freqs(self, calc):
        imag_freqs = []
        normal_mode_section = False

        for line in calc.output.file_lines:
            if 'normal coordinates' in line:
                normal_mode_section = True
                imag_freqs = []

            if 'Thermochemistry' in line:
                normal_mode_section = False

            if normal_mode_section and 'Frequencies' in line:
                freqs = [float(line.split()[i])
                         for i in range(2, len(line.split()))]
                for freq in freqs:
                    if freq < 0:
                        imag_freqs.append(freq)

        logger.info(f'Found imaginary freqs {imag_freqs}')
        return imag_freqs

    def get_normal_mode_displacements(self, calc, mode_number):
        # mode numbers start at 1, not 6
        mode_number -= 5
        start_col = 0
        normal_mode_section, displacements = False, []
        correct_mode_section = False

        for j, line in enumerate(calc.output.file_lines):
            if 'normal coordinates' in line:
                normal_mode_section = True
                displacements = []

            if 'Thermochemistry' in line:
                normal_mode_section = False

            if correct_mode_section and len(line.split()) > 3 and line.split()[0].isdigit():
                displacements.append([float(line.split()[k]) for k in range(start_col, start_col + 3)])

            if normal_mode_section and len(line.split()) == 3 and line.split()[0].isdigit():
                mode_numbers = [int(n) for n in line.split()]
                if mode_number in mode_numbers:
                    correct_mode_section = True
                    start_col = 3 * [i for i in range(len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 2
                else:
                    correct_mode_section = False

        return np.array(displacements)

    def get_final_atoms(self, calc):

        atoms = None

        for i, line in enumerate(calc.output.file_lines):

            if 'Standard orientation' in line or 'Input orientation' in line:

                atoms = []
                xyz_lines = calc.output.file_lines[i+5:i+5+calc.molecule.n_atoms]

                for xyz_line in xyz_lines:
                    atom_index, _, _, x, y, z = xyz_line.split()
                    atom_index = int(atom_index) - 1
                    atoms.append(Atom(calc.molecule.atoms[atom_index].label, x=x, y=y, z=z))

                if len(atoms) != calc.molecule.n_atoms:
                    raise AtomsNotFound

        if atoms is None:
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self, calc):

        charges_section = False
        charges = []
        for line in reversed(calc.output.file_lines):
            if 'sum of mulliken charges' in line.lower():
                charges_section = True

            if len(charges) == calc.molecule.n_atoms:
                return list(reversed(charges))

            if charges_section and len(line.split()) == 3:
                charges.append(float(line.split()[2]))

        logger.error('Something went wrong finding the atomic charges')
        return None

    def get_gradients(self, calc):
        gradients_section = False
        gradients = []
        dashed_line = 0

        for line in calc.output.file_lines:

            if 'Axes restored to original set' in line:
                gradients_section = True
                gradients = []
                dashed_line = 0

            if gradients_section and '--------' in line:
                dashed_line += 1
                if dashed_line == 3:
                    gradients_section = False

            if gradients_section and len(line.split()) == 5:
                _, _, fx, fy, fz = line.split()
                try:
                    # Ha / a0
                    force = np.array([float(fx), float(fy), float(fz)])

                    grad = -force / Constants.a02ang
                    gradients.append(grad)
                except ValueError:
                    pass
        for line in gradients:
            for i in range(3):
                line[i] *= -1

        return np.array(gradients)

    def __init__(self, name='g09', path=None, keywords_set=None,
                 implicit_solvation_type=None):
        """Gaussian 09"""

        if keywords_set is None:
            keywords_set = Config.G09.keywords

        if implicit_solvation_type is None:
            implicit_solvation_type = Config.G09.implicit_solvation_type

        super().__init__(name=name,
                         path=Config.G09.path if path is None else path,
                         keywords_set=keywords_set,
                         implicit_solvation_type=implicit_solvation_type,
                         doi='http://gaussian.com/citation/')


g09 = G09()
