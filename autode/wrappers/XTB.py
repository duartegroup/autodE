from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.input_output import xyzs2xyzfile
from autode.wrappers.base import ElectronicStructureMethod
import numpy as np
from autode.wrappers.base import req_methods


solvents = ['Acetone', 'Acetonitrile', 'Benzene', 'CH2Cl2', 'CHCl3', 'CS2', 'DMF', 'DMSO', 'Ether', 'Water',
            'H2O', 'Methanol', 'n-Hexane', 'THF', 'Toluene']

XTB = ElectronicStructureMethod(name='xtb',
                                path=Config.XTB.path,
                                aval_solvents=[solv.lower() for solv in solvents])


def generate_input(calc):

    calc.input_filename = calc.name + '_xtb.xyz'
    xyzs2xyzfile(calc.xyzs, filename=calc.input_filename)
    calc.output_filename = calc.name + '_xtb.out'

    # Add
    calc.flags = ['--chrg', str(calc.charge)]

    if calc.opt:
        calc.flags.append('--opt')

    if calc.solvent:
        calc.flags += ['--gbsa', calc.solvent]

    if calc.distance_constraints:
        xcontrol_filename = 'xcontrol_' + calc.name
        with open(xcontrol_filename, 'w') as xcontrol_file:
            for atom_ids in calc.distance_constraints.keys():  # XTB counts from 1 so increment atom ids by 1
                print('$constrain\nforce constant=10\ndistance:' + str(atom_ids[0] + 1) + ', ' + str(
                    atom_ids[1] + 1) + ', ' + str(np.round(calc.distance_constraints[atom_ids], 3)) + '\n$',
                      file=xcontrol_file)

        calc.flags += ['--input', xcontrol_filename]

    return None


def calculation_terminated_normally(calc):

    for n_line, line in enumerate(calc.rev_output_file_lines):
        if 'ERROR' in line:
            return False
        if n_line > 20:
            # With XTB we will search for there being no '#ERROR!' in the last few lines
            return True


def get_energy(calc):
    for line in calc.rev_output_file_lines:
        if 'total E' in line:
            return float(line.split()[-1])
        if 'TOTAL ENERGY' in line:
            return float(line.split()[-3])


def optimisation_converged(calc):

    for line in calc.rev_output_file_lines:
        if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
            return True

    return False


def optimisation_nearly_converged(calc):
    raise NotImplementedError


def get_imag_freqs(calc):
    raise NotImplementedError


def get_normal_mode_displacements(calc, mode_number):
    raise NotImplementedError


def get_final_xyzs(calc):
    xyzs = []

    geom_section = False
    for line in calc.output_file_lines:
        if '$coord' in line:
            geom_section = True

        if '$end' in line and geom_section:
            geom_section = False

        if len(line.split()) == 4 and geom_section:
            x, y, z, atom_label = line.split()
            xyzs.append([atom_label, float(x) * Constants.a02ang,
                         float(y) * Constants.a02ang, float(z) * Constants.a02ang])

    return xyzs


# Bind all the required functions to the class definition
[setattr(XTB, method, globals()[method]) for method in req_methods]
