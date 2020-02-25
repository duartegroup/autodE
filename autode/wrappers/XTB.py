from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.input_output import xyzs2xyzfile
from autode.wrappers.base import ElectronicStructureMethod
import numpy as np
from autode.wrappers.base import req_methods
import os


XTB = ElectronicStructureMethod(name='xtb', path=Config.XTB.path)

XTB.__name__ = 'XTB'


def generate_input(calc):

    calc.input_filename = calc.name + '_xtb.xyz'
    xyzs2xyzfile(calc.xyzs, filename=calc.input_filename)
    calc.output_filename = calc.name + '_xtb.out'

    # Add
    calc.flags = ['--chrg', str(calc.charge)]

    if calc.opt:
        calc.flags.append('--opt')

    if calc.grad:
        calc.flags.append('--grad')

    if calc.solvent_keyword:
        calc.flags += ['--gbsa', calc.solvent_keyword]

    if calc.distance_constraints or calc.cartesian_constraints or calc.charges:
        force_constant = 10
        if calc.constraints_already_met:
            force_constant += 90
        xcontrol_filename = 'xcontrol_' + calc.name
        with open(xcontrol_filename, 'w') as xcontrol_file:
            if calc.distance_constraints:
                for atom_ids in calc.distance_constraints.keys():  # XTB counts from 1 so increment atom ids by 1
                    print(f'$constrain\nforce constant={force_constant}\ndistance:' + str(atom_ids[0] + 1) + ', ' + str(
                        atom_ids[1] + 1) + ', ' + str(np.round(calc.distance_constraints[atom_ids], 3)) + '\n$',
                        file=xcontrol_file)

            if calc.cartesian_constraints:
                constrained_atoms = [i + 1 for i in calc.cartesian_constraints]
                list_of_ranges = []
                used_atoms = []
                for atom in constrained_atoms:
                    rang = []
                    if atom not in used_atoms:
                        while atom in constrained_atoms:
                            used_atoms.append(atom)
                            rang.append(atom)
                            atom += 1
                        if len(rang) in (1, 2):
                            list_of_ranges += rang
                        else:
                            range_string = str(rang[0]) + '-' + str(rang[-1])
                            list_of_ranges.append(range_string)
                print('$constrain\nforce constant=100\natoms:',
                      end=' ', file=xcontrol_file)
                print(*list_of_ranges, sep=',', file=xcontrol_file)
                print('$', file=xcontrol_file)

            if calc.charges:
                print(f'$embedding\ninput={calc.name}_xtb.pc\ninput=orca\n$end', file=xcontrol_file)

        calc.flags += ['--input', xcontrol_filename]
        calc.additional_input_files.append(xcontrol_filename)

    if calc.charges:
        with open(f'{calc.name}_xtb.pc', 'w') as pc_file:
            print(len(calc.charges), file=pc_file)
            for line in calc.charges:
                formatted_line = [line[-1]] + line[1:4] + [line[0]]
                print('{:^12.8f} {:^12.8f} {:^12.8f} {:^12.8f} {:<3}'.format(*formatted_line), file=pc_file)
        calc.additional_input_files.append(f'{calc.name}_xtb.pc')

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


def get_atomic_charges(calc):
    charges_sect = False
    charges = []
    for line in calc.output_file_lines:
        if 'Mol.' in line:
            charges_sect = False
        if charges_sect and len(line.split()) == 7:
            charges.append(float(line.split()[4]))
        if 'covCN' in line:
            charges_sect = True
    return charges


def get_gradients(calc):
    gradients = []
    if os.path.exists(f'{calc.name}_xtb.grad'):
        grad_file_name = f'{calc.name}_xtb.grad'
        with open(grad_file_name, 'r') as grad_file:
            for line in grad_file:
                x, y, z = line.split()
                gradients.append([float(x), float(y), float(z)])
    else:
        with open('gradient', 'r') as grad_file:
            for line_no, line in enumerate(grad_file):
                if line_no > 1 and len(line.split()) == 3:
                    x, y, z = line.split()
                    gradients.append([float(x.replace('D', 'E')), float(y.replace('D', 'E')), float(z.replace('D', 'E'))])
        with open(f'{calc.name}_xtb.grad', 'w') as new_grad_file:
            [print('{:^12.8f} {:^12.8f} {:^12.8f}'.format(*line), file=new_grad_file) for line in gradients]
        os.remove('gradient')
    return gradients


# Bind all the required functions to the class definition
[setattr(XTB, method, globals()[method]) for method in req_methods]
