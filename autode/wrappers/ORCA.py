import numpy as np
import os
from autode.constants import Constants
from autode.utils import run_external
from autode.wrappers.base import ElectronicStructureMethod
from autode.atoms import Atom, get_atomic_weight
from autode.config import Config
from autode.exceptions import UnsuppportedCalculationInput
from autode.exceptions import NoCalculationOutput
from autode.utils import work_in_tmp_dir
from autode.log import logger

vdw_gaussian_solvent_dict = {'water': 'Water', 'acetone': 'Acetone', 'acetonitrile': 'Acetonitrile', 'benzene': 'Benzene',
                             'carbon tetrachloride': 'CCl4', 'dichloromethane': 'CH2Cl2', 'chloroform': 'Chloroform', 'cyclohexane': 'Cyclohexane',
                             'n,n-dimethylformamide': 'DMF', 'dimethylsulfoxide': 'DMSO', 'ethanol': 'Ethanol', 'n-hexane': 'Hexane',
                             'methanol': 'Methanol', '1-octanol': 'Octanol', 'pyridine': 'Pyridine', 'tetrahydrofuran': 'THF', 'toluene': 'Toluene'}


def use_vdw_gaussian_solvent(keywords, implicit_solv_type):
    """
    Determine if the calculation should use the gaussian charge scheme which
    generally affords better convergence for optimiations in implicit solvent

    Arguments:
        keywords (autode.wrappers.keywords.Keywords):
        implicit_solv_type (str):

    Returns:
        (bool):
    """
    if implicit_solv_type.lower() != 'cpcm':
        return False

    if any('freq' in kw.lower() or 'optts' in kw.lower() for kw in keywords):
        logger.warning('Cannot do analytical frequencies with gaussian charge '
                       'scheme - switching off')
        return False

    return True


def add_solvent_keyword(calc_input, keywords, implicit_solv_type):
    """Add a keyword to the input file based on the solvent"""

    if implicit_solv_type.lower() not in ['smd', 'cpcm']:
        raise UnsuppportedCalculationInput

    # Use CPCM solvation
    if (use_vdw_gaussian_solvent(keywords, implicit_solv_type)
            and calc_input.solvent not in vdw_gaussian_solvent_dict.keys()):

        err = (f'CPCM solvent with gaussian charge not avalible for '
               f'{calc_input.solvent}.Available solvents are '
               f'{vdw_gaussian_solvent_dict.keys()}')

        raise UnsuppportedCalculationInput(message=err)

    keywords.append(f'CPCM({vdw_gaussian_solvent_dict[calc_input.solvent]})')
    return


def get_keywords(calc_input, molecule, implicit_solv_type):
    """Modify the keywords for this calculation with the solvent + fix for
    single atom optimisation calls"""

    keywords = calc_input.keywords.copy()

    for keyword in keywords:
        if 'opt' in keyword.lower() and molecule.n_atoms == 1:
            logger.warning('Can\'t optimise a single atom')
            keywords.remove(keyword)  # ORCA defaults to a SP calc

    if calc_input.solvent is not None:
        add_solvent_keyword(calc_input, keywords, implicit_solv_type)

    return keywords


def print_solvent(inp_file, calc_input, keywords, implicit_solv_type):
    """Add the solvent block to the input file"""
    if calc_input.solvent is None:
        return

    if implicit_solv_type.lower() == 'smd':
        print(f'%cpcm\n'
              f'smd true\n'
              f'SMDsolvent \"{calc_input.solvent}\"\n'
              f'end', file=inp_file)

    if use_vdw_gaussian_solvent(keywords, implicit_solv_type):
        print('%cpcm\n surfacetype vdw_gaussian\nend', file=inp_file)

    return


def print_added_internals(inp_file, calc_input):
    """Print the added internal coordinates"""

    if calc_input.added_internals is None:
        return

    for (i, j) in calc_input.added_internals:
        print('%geom\n'
              'modify_internal\n'
              '{ B', i, j, 'A } end\n'
              'end', file=inp_file)
    return


def print_distance_constraints(inp_file, molecule):
    """Print the distance constraints to the input file"""
    if molecule.constraints.distance is None:
        return

    print('%geom Constraints', file=inp_file)
    for (i, j), dist in molecule.constraints.distance.items():
        print('{ B', i, j, dist, 'C }', file=inp_file)
    print('    end\nend', file=inp_file)

    return


def print_cartesian_constraints(inp_file, molecule):
    """Print the Cartesian constraints to the input file"""

    if molecule.constraints.cartesian is None:
        return

    print('%geom Constraints', file=inp_file)
    for i in molecule.constraints.cartesian:
        print('{ C', i, 'C }', file=inp_file)
    print('    end\nend', file=inp_file)

    return


def print_increased_optimisation_steps(inp_file, molecule, calc_input):
    """If there are relatively few atoms increase the number of opt steps"""

    if molecule.n_atoms > 33:
        return

    block = calc_input.other_block
    if block is None or 'maxit' not in block.lower():
        print('%geom MaxIter 100 end', file=inp_file)

    return


def print_point_charges(inp_file, calc_input):
    """Print a point charge file and add the name to the input file"""

    if calc_input.point_charges is None:
        return

    filename = calc_input.filename.replace('.inp', '.pc')
    with open(filename, 'w') as pc_file:
        print(len(calc_input.point_charges), file=pc_file)
        for pc in calc_input.point_charges:
            x, y, z = pc.coord
            print(f'{pc.charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
                  file=pc_file)

    calc_input.additional_filenames.append(filename)

    print(f'% pointcharges "{filename}"', file=inp_file)
    return


def print_default_params(inp_file):
    """Print some useful default parameters to the input file"""

    print('%output \nxyzfile=True \nend ',
          '%scf \nmaxiter 250 \nend',
          '%output\nPrint[P_Hirshfeld] = 1\nend',
          '% maxcore', Config.max_core, sep='\n', file=inp_file)
    return


def print_coordinates(inp_file, molecule):
    """Print the coordinates to the input file in the correct format"""

    print('*xyz', molecule.charge, molecule.mult, file=inp_file)
    for atom in molecule.atoms:
        x, y, z = atom.coord
        print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
              file=inp_file)
    print('*', file=inp_file)

    return


def calc_atom_entropy(atom_label, temp):
    """
    Calculate the entropy of a single atom, not available in ORCA as only the
    translational contribution: S_trans = R (ln(q_trans) + 5R/2)

    Arguments:
        atom_label (str):
        temp (float): Temperature in K

    Returns:
        (float):
    """

    k_b = 1.38064852E-23          # J K-1
    h = 6.62607004E-34            # J s
    n_a = 6.022140857E23          # molecules mol-1
    atm_to_pa = 101325            # Pa
    amu_to_kg = 1.660539040E-27   # Kg

    mass = amu_to_kg * get_atomic_weight(atom_label=atom_label)
    v_eff = k_b * temp / atm_to_pa
    q_trans = ((2.0 * np.pi * mass * k_b * temp / h**2)**1.5 * v_eff)

    s = k_b * n_a * (np.log(q_trans) + 2.5)
    # Convert from J K-1 mol-1 to K-1 Ha
    return s / (Constants.ha2kJmol * 1000)


class ORCA(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):

        keywords = get_keywords(calc.input, molecule,
                                self.implicit_solvation_type)

        with open(calc.input.filename, 'w') as inp_file:
            print('!', *keywords, file=inp_file)

            print_solvent(inp_file, calc.input, keywords,
                          self.implicit_solvation_type)
            print_added_internals(inp_file, calc.input)
            print_distance_constraints(inp_file, molecule)
            print_cartesian_constraints(inp_file, molecule)
            print_increased_optimisation_steps(inp_file, molecule, calc.input)
            print_point_charges(inp_file, calc.input)
            print_default_params(inp_file)

            if calc.input.other_block is not None:
                print(calc.input.other_block, file=inp_file)

            if calc.n_cores > 1:
                print(f'%pal nprocs {calc.n_cores}\nend', file=inp_file)

            if calc.input.temp is not None:
                print(f'%freq  Temp {calc.input.temp}\nend', file=inp_file)

            print_coordinates(inp_file, molecule)

        return None

    def get_input_filename(self, calculation):
        return f'{calculation.name}.inp'

    def get_output_filename(self, calculation):
        return f'{calculation.name}.out'

    def execute(self, calc):

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.out', '.hess', '.xyz', '.inp', '.pc'))
        def execute_orca():
            run_external(params=[calc.method.path, calc.input.filename],
                         output_filename=calc.output.filename)

        execute_orca()
        return None

    def calculation_terminated_normally(self, calc):

        termination_strings = ['ORCA TERMINATED NORMALLY',
                               'The optimization did not converge']

        for n_line, line in enumerate(reversed(calc.output.file_lines)):

            if any(substring in line for substring in termination_strings):
                logger.info('orca terminated normally')
                return True

            if n_line > 30:
                # The above lines are pretty close to the end of the file –
                # so skip parsing it all
                return False

        return False

    def get_energy(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'FINAL SINGLE POINT ENERGY' in line:
                return float(line.split()[4])

        return None

    def get_enthalpy(self, calc):
        """Get the enthalpy (H) from an ORCA calculation output"""

        for line in reversed(calc.output.file_lines):
            if 'Total Enthalpy' in line:

                try:
                    return float(line.split()[-2])

                except ValueError:
                    break

        logger.error('Could not get the free energy from the calculation. '
                     'Was a frequency requested?')
        return None

    def get_free_energy(self, calc):
        """Get the Gibbs free energy (G) from an ORCA calculation output"""

        if calc.molecule.n_atoms == 1:
            logger.warning('ORCA fails to calculate the entropy for a single '
                           'atom, returning the correct G in 1 atm')
            h = self.get_enthalpy(calc)
            s = calc_atom_entropy(atom_label=calc.molecule.atoms[0].label,
                                  temp=calc.input.temp)  # J K-1 mol-1

            # Calculate H - TS, the latter term from Jmol-1 -> Ha
            return h - s * calc.input.temp

        for line in reversed(calc.output.file_lines):
            if ('Final Gibbs free energy' in line
                    or 'Final Gibbs free enthalpy' in line):

                try:
                    return float(line.split()[-2])

                except ValueError:
                    break

        logger.error('Could not get the free energy from the calculation. '
                     'Was a frequency requested?')
        return None

    def optimisation_converged(self, calc):

        for line in reversed(calc.output.file_lines):
            if 'THE OPTIMIZATION HAS CONVERGED' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        geom_conv_block = False

        for line in reversed(calc.output.file_lines):
            if geom_conv_block and 'Geometry convergence' in line:
                geom_conv_block = False
            if 'The optimization has not yet converged' in line:
                geom_conv_block = True
            if geom_conv_block and len(line.split()) == 5:
                if line.split()[-1] == 'YES':
                    return True

        return False

    def get_imaginary_freqs(self, calc):
        imag_freqs = []

        for i, line in enumerate(calc.output.file_lines):
            if 'VIBRATIONAL FREQUENCIES' in line:
                last_line = i + 3 * calc.molecule.n_atoms + 5

                # Reset every time freqs are found, so the final is returned
                freq_lines = calc.output.file_lines[i + 5:last_line]
                freqs = [float(l.split()[1]) for l in freq_lines]
                imag_freqs = [freq for freq in freqs if freq < 0]

        logger.info(f'Found imaginary freqs {imag_freqs}')
        return imag_freqs

    def get_normal_mode_displacements(self, calc, mode_number):
        normal_mode_section, values_sec, = False, False
        displacements, col = [], None

        for j, line in enumerate(calc.output.file_lines):
            if 'NORMAL MODES' in line:
                normal_mode_section, values_sec, = True, False
                displacements, col = [], None

            if 'IR SPECTRUM' in line:
                normal_mode_section, values_sec = False, False

            if normal_mode_section and len(line.split()) > 1:
                if line.split()[0].startswith('0'):
                    values_sec = True

            if not values_sec:
                continue

            if '.' in line or len(line.split()) < 2:
                continue

            mode_numbers = [int(val) for val in line.split()]
            if mode_number not in mode_numbers:
                continue

            col = [i for i in range(len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 1

            d_lines = calc.output.file_lines[j+1:j+3 * calc.molecule.n_atoms+1]
            displacements = [float(d_line.split()[col]) for d_line in d_lines]

        displacements_xyz = [displacements[i:i + 3] for i in range(0, len(displacements), 3)]

        return np.array(displacements_xyz)

    def get_final_atoms(self, calc):

        atoms = []
        xyz_file_name = calc.output.filename.replace('.out', '.xyz')

        if not os.path.exists(xyz_file_name):
            raise NoCalculationOutput

        with open(xyz_file_name, 'r') as xyz_file:
            for line_no, line in enumerate(xyz_file):
                if line_no > 1:
                    atom_label, x, y, z = line.split()
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))

        return atoms

    def get_atomic_charges(self, calc):
        """
        e.g.

       .HIRSHFELD ANALYSIS
        ------------------

        Total integrated alpha density =     12.997461186
        Total integrated beta density  =     12.997461186

          ATOM     CHARGE      SPIN
           0 C   -0.006954    0.000000
           . .      .            .
        """
        charges = []

        for i, line in enumerate(calc.output.file_lines):
            if 'HIRSHFELD ANALYSIS' in line:
                charges = []
                first, last = i+7, i+7+calc.molecule.n_atoms
                for charge_line in calc.output.file_lines[first:last]:
                    charges.append(float(charge_line.split()[-1]))

        return charges

    def get_gradients(self, calc):
        """
        e.g.

        #------------------
        CARTESIAN GRADIENT                                            <- i
        #------------------

           1   C   :   -0.011390275   -0.000447412    0.000552736    <- j
        """
        gradients = []

        for i, line in enumerate(calc.output.file_lines):
            if 'CARTESIAN GRADIENT' in line:
                gradients = []
                first, last = i + 3, i + 3 + calc.molecule.n_atoms
                for grad_line in calc.output.file_lines[first:last]:
                    dadx, dady, dadz = grad_line.split()[-3:]
                    vec = [float(dadx), float(dady), float(dadz)]

                    gradients.append(np.array(vec))

        # Convert from Ha a0^-1 to Ha A-1
        gradients = [grad / Constants.a02ang for grad in gradients]
        return np.array(gradients)

    def __init__(self):
        super().__init__('orca', path=Config.ORCA.path,
                         keywords_set=Config.ORCA.keywords,
                         implicit_solvation_type=Config.ORCA.implicit_solvation_type)


orca = ORCA()
