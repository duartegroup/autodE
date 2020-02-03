from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods
from autode.geom import coords2xyzs
from autode.input_output import xyzs2xyzfile
from copy import deepcopy
import numpy as np
import os

G09 = ElectronicStructureMethod(name='g09', path=Config.G09.path,
                                scan_keywords=Config.G09.scan_keywords,
                                conf_opt_keywords=Config.G09.conf_opt_keywords,
                                opt_keywords=Config.G09.opt_keywords,
                                opt_ts_keywords=Config.G09.opt_ts_keywords,
                                hess_keywords=Config.G09.hess_keywords,
                                sp_keywords=Config.G09.sp_keywords)

G09.__name__ = 'G09'


def generate_input(calc):
    calc.input_filename = calc.name + '_g09.com'
    calc.output_filename = calc.name + '_g09.log'
    keywords = calc.keywords.copy()

    if calc.distance_constraints or calc.cartesian_constraints or calc.bond_ids_to_add:
        keywords.append('Geom=ModRedun')

    if calc.n_atoms == 1:
        for keyword in keywords:
            if 'opt' in keyword.lower():
                logger.warning('Cannot do an optimisation for a single atom')
                keywords.remove(keyword)

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

        print(calc.charge, calc.mult, file=inp_file)
        [print('{:<3} {:^12.8f} {:^12.8f} {:^12.8f}'.format(*line), file=inp_file) for line in calc.xyzs]

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


def calculation_terminated_normally(calc):

    for line in calc.rev_output_file_lines:

        if 'Normal termination of Gaussian' in line or 'Number of steps exceeded' in line:
            logger.info('Gaussian09 terminated normally')
            return True

        if 'Bend failed for angle' in line:
            logger.info('Gaussian encountered a 180° angle and crashed, using cartesian coordinates in the optimisation for a few cycles')
            cart_calc = deepcopy(calc)
            for keyword in cart_calc.keywords:
                if keyword.lower().startswith('geom'):
                    cart_calc.keywords.remove(keyword)
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
                    cart_calc.keywords.remove(keyword)
                    cart_calc.keywords.append(new_keyword)

            cart_calc.name += '_cartesian'
            cart_calc.xyzs = calc.get_final_xyzs()
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
            fixed_angle_calc.xyzs = cart_calc.get_final_xyzs()
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


def get_energy(calc):
    for line in calc.rev_output_file_lines:
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


def optimisation_converged(calc):
    for line in calc.rev_output_file_lines:
        if 'Optimization completed' in line:
            return True

    return False


def optimisation_nearly_converged(calc):
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


def get_imag_freqs(calc):
    imag_freqs = None
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


def get_normal_mode_displacements(calc, mode_number):
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
                        for i in range(calc.n_atoms):
                            disp_line = calc.output_file_lines[j + 7 + i]
                            xyz_disp = [float(disp_line.split()[k])
                                        for k in range(start_col, start_col + 3)]
                            displacements.append(xyz_disp)
                except ValueError:
                    pass

    if len(displacements) != calc.n_atoms:
        logger.error('Something went wrong getting the displacements n != n_atoms')
        return None

    return displacements


def get_final_xyzs(calc):

    coords = np.zeros((calc.n_atoms, 3))
    xyz_section = False
    dashed_line = 0

    for line in calc.output_file_lines:

        if 'Standard orientation' in line:
            xyz_section = True
            dashed_line = 0

        if xyz_section and '--------' in line:
            dashed_line += 1
            if dashed_line == 3:
                xyz_section = False

                if atom_index != calc.n_atoms - 1:
                    logger.critical('Calc changed the number of atoms')
                    exit()

        if xyz_section and len(line.split()) == 6:
            atom_index, _, _, x, y, z = line.split()
            try:
                atom_index = int(atom_index) - 1
                coords[atom_index][0] = float(x)
                coords[atom_index][1] = float(y)
                coords[atom_index][2] = float(z)
            except ValueError:
                pass

    xyzs = coords2xyzs(coords, calc.xyzs)

    zero_xyzs = False
    for xyz in xyzs:
        if all(xyz[i] == 0.0 for i in range(1, 4)):
            if zero_xyzs:
                return []
            else:
                zero_xyzs = True

    xyz_filename = f'{calc.name}_g09.xyz'
    xyzs2xyzfile(xyzs, xyz_filename)

    return xyzs


# Bind all the required functions to the class definition
[setattr(G09, method, globals()[method]) for method in req_methods]
