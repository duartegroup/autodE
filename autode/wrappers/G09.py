from copy import deepcopy
import numpy as np
from autode.wrappers.base import ElectronicStructureMethod
from autode.atoms import Atom
from autode.config import Config
from autode.exceptions import AtomsNotFound
from autode.log import logger


def modify_keywords_for_point_charges(keywords):
    """For a list of Gaussian keywords modify to include z-matrix if not already included"""
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


class G09(ElectronicStructureMethod):
    # TODO implement partial hessian

    def generate_input(self, calc):
        calc.input_filename = calc.name + '_g09.com'
        calc.output_filename = calc.name + '_g09.log'
        keywords = calc.keywords_list.copy()

        if calc.distance_constraints or calc.cartesian_constraints or calc.bond_ids_to_add:
            keywords.append('Geom=ModRedun')

        for keyword in keywords:

            if 'opt' in keyword.lower():
                calc.opt = True

                if calc.n_atoms == 1:
                    logger.warning('Cannot do an optimisation for a single atom')
                    keywords.remove(keyword)

        if calc.point_charges:
            modify_keywords_for_point_charges(keywords)

        # By default perform all optimisations without symmetry
        if calc.opt and not any(k.lower() == 'nosymm' for k in keywords):
            keywords.append('NoSymm')

        with open(calc.input_filename, 'w') as inp_file:
            print(f'%mem={calc.max_core_mb}MB', file=inp_file)
            if calc.n_cores > 1:
                print(f'%nprocshared={calc.n_cores}', file=inp_file)

            print('#', *keywords, file=inp_file, end=' ')

            if calc.solvent_keyword:
                print(f'scrf=(smd,solvent={calc.solvent_keyword})', file=inp_file)
            else:
                print('', file=inp_file)

            print(f'\n {calc.name}\n', file=inp_file)

            print(calc.molecule.charge, calc.molecule.mult, file=inp_file)

            for atom in calc.molecule.atoms:
                print(f'{atom.label:<3} {atom.coord[0]:^12.8f} {atom.coord[1]:^12.8f} {atom.coord[2]:^12.8f}',
                      file=inp_file)

            if calc.point_charges is not None:
                print('', file=inp_file)
                for point_charge in calc.point_charges:
                    x, y, z = point_charge.coord
                    print(f'{x:^12.8f} {y:^12.8f} {z:^12.8f} {point_charge.charge:^12.8f}', file=inp_file)

            print('', file=inp_file)

            if calc.bond_ids_to_add:
                for bond_ids in calc.bond_ids_to_add:
                    print('B', bond_ids[0] + 1, bond_ids[1] + 1, file=inp_file)

            if calc.distance_constraints:
                for bond_ids in calc.distance_constraints.keys():  # Gaussian counts from 1
                    print('B', bond_ids[0]+1, bond_ids[1]+1,
                          calc.distance_constraints[bond_ids], 'B', file=inp_file)
                    print('B', bond_ids[0]+1, bond_ids[1]+1, 'F', file=inp_file)

            if calc.cartesian_constraints:
                [print('X', atom_id+1, 'F', file=inp_file)
                 for atom_id in calc.cartesian_constraints]

        return None

    def calculation_terminated_normally(self, calc):

        for line in calc.rev_output_file_lines:

            if 'Normal termination of Gaussian' in line or 'Number of steps exceeded' in line:
                logger.info('Gaussian09 terminated normally')
                return True

            if 'Bend failed for angle' in line:
                logger.info('Gaussian encountered a 180° angle and crashed, using cartesian coordinates in the optimisation for a few cycles')
                cart_calc = deepcopy(calc)
                for keyword in cart_calc.keywords_list.copy():
                    if keyword.lower().startswith('geom'):
                        cart_calc.keywords_list.remove(keyword)
                    elif keyword.lower().startswith('opt'):
                        options = []
                        if '=(' in keyword:
                            # get the individual options
                            messy_options = keyword[5:-1].split(',')
                            options = [option.lower().strip()
                                       for option in messy_options]
                            for option in options:
                                if option.startswith('maxcycles') or option.startswith('maxstep'):
                                    options.remove(option)
                        elif '=' in keyword:
                            options = [keyword[4:]]
                        options.append('maxcycles=3')
                        options.append('maxstep=1')
                        options.append('cartesian')
                        new_keyword = 'Opt=('
                        new_keyword += ', '.join(options)
                        new_keyword += ')'
                        cart_calc.keywords_list.remove(keyword)
                        cart_calc.keywords_list.append(new_keyword)

                cart_calc.name += '_cartesian'
                cart_calc.xyzs = calc.get_final_atoms()
                cart_calc.distance_constraints = None
                cart_calc.cartesian_constraints = None
                cart_calc.bond_ids_to_add = None
                cart_calc.input_filename = None
                cart_calc.output_filename = None
                cart_calc.output_file_exists = False
                cart_calc.terminated_normally = False
                cart_calc.output_file_lines = None
                cart_calc.rev_output_file_lines = None
                cart_calc.run()
                for line in cart_calc.rev_output_file_lines:
                    if 'Normal termination of Gaussian' in line:
                        calc.name = cart_calc.name
                        calc.output_filename = cart_calc.output_filename
                        calc.set_output_file_lines()
                        logger.info('The cartesian optimisation converged')
                        return True

                logger.info('Returning to internal coordinates')

                fixed_angle_calc = deepcopy(calc)
                fixed_angle_calc.name += '_internal'
                fixed_angle_calc.xyzs = cart_calc.get_final_atoms()
                fixed_angle_calc.input_filename = None
                fixed_angle_calc.output_filename = None
                fixed_angle_calc.output_file_exists = False
                fixed_angle_calc.terminated_normally = False
                fixed_angle_calc.output_file_lines = None
                fixed_angle_calc.rev_output_file_lines = None
                fixed_angle_calc.run()
                if fixed_angle_calc.terminated_normally:
                    logger.info('The 180° angle issue has been fixed')
                    calc.output_filename = fixed_angle_calc.output_filename
                    calc.name = fixed_angle_calc.name
                    calc.set_output_file_lines()
                    return True
                else:
                    return False

        return False

    def get_energy(self, calc):
        for line in calc.rev_output_file_lines:
            # if 'Sum of electronic and thermal Enthalpies' in line:
            #     return float(line.split()[-1])
            if 'Sum of electronic and thermal Free Energies' in line:
                return float(line.split()[-1])
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

    def optimisation_converged(self, calc):
        for line in calc.rev_output_file_lines:
            if 'Optimization completed' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        geom_conv_block = False

        for line in calc.rev_output_file_lines:
            if geom_conv_block and 'Item' in line:
                geom_conv_block = False
            if 'Predicted change in Energy' in line:
                geom_conv_block = True
            if geom_conv_block and len(line.split()) == 4:
                if line.split()[-1] == 'YES':
                    return True
        return False

    def get_imag_freqs(self, calc):
        imag_freqs = []
        normal_mode_section = False

        for line in calc.output_file_lines:
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
        normal_mode_section, displacements = False, []

        for j, line in enumerate(calc.output_file_lines):
            if 'normal coordinates' in line:
                normal_mode_section = True
                displacements = []

            if 'Thermochemistry' in line:
                normal_mode_section = False

            if normal_mode_section:
                if len(line.split()) == 3:
                    try:
                        mode_numbers = [int(val) for val in line.split()]
                        if mode_number in mode_numbers:
                            start_col = 3 * [i for i in range(len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 2
                            for i in range(calc.molecule.n_atoms):
                                disp_line = calc.output_file_lines[j + 7 + i]
                                xyz_disp = [float(disp_line.split()[k])
                                            for k in range(start_col, start_col + 3)]
                                displacements.append(xyz_disp)
                    except ValueError:
                        pass

        if len(displacements) != calc.molecule.n_atoms:
            logger.error('Something went wrong getting the displacements n != n_atoms')
            return None

        return np.array(displacements)

    def get_final_atoms(self, calc):

        atoms = None

        for i, line in enumerate(calc.output_file_lines):

            if 'Standard orientation' in line or 'Input orientation' in line:

                atoms = []
                xyz_lines = calc.output_file_lines[i+5:i+5+calc.n_atoms]

                for xyz_line in xyz_lines:
                    atom_index, _, _, x, y, z = xyz_line.split()
                    atom_index = int(atom_index) - 1
                    atoms.append(Atom(calc.molecule.atoms[atom_index].label, x=x, y=y, z=z))

                if len(atoms) != calc.n_atoms:
                    raise AtomsNotFound

        if atoms is None:
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self, calc):

        charges_section = False
        charges = []
        for line in calc.rev_output_file_lines:
            if 'sum of mulliken charges' in line.lower():
                charges_section = True

            if len(charges) == calc.n_atoms:
                return list(reversed(charges))

            if charges_section and len(line.split()) == 3:
                charges.append(float(line.split()[2]))

        logger.error('Something went wrong finding the atomic charges')
        return None

    def get_gradients(self, calc):
        gradients_section = False
        gradients = []
        dashed_line = 0

        for line in calc.output_file_lines:

            if 'Axes restored to original set' in line:
                gradients_section = True
                gradients = []
                dashed_line = 0

            if gradients_section and '--------' in line:
                dashed_line += 1
                if dashed_line == 3:
                    gradients_section = False

            if gradients_section and len(line.split()) == 5:
                _, _, x, y, z = line.split()
                try:
                    gradients.append([float(x), float(y), float(z)])
                except ValueError:
                    pass
        for line in gradients:
            for i in range(3):
                line[i] *= -1

        return gradients

    def __init__(self):
        super().__init__(name='g09', path=Config.G09.path, keywords=Config.G09.keywords)


g09 = G09()
