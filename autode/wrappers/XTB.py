import numpy as np
import os
from autode.wrappers.base import ElectronicStructureMethod
from autode.atoms import Atom
from autode.config import Config
from autode.constants import Constants
from autode.exceptions import AtomsNotFound


class XTB(ElectronicStructureMethod):

    def generate_input(self, calc):

        calc.input_filename = calc.name + '_xtb.xyz'

        calc.molecule.print_xyz_file(filename=calc.input_filename)

        calc.output_filename = calc.name + '_xtb.out'

        calc.flags = ['--chrg', str(calc.molecule.charge)]

        if calc.opt:
            calc.flags.append('--opt')

        if calc.grad:
            calc.flags.append('--grad')

        if calc.solvent_keyword:
            calc.flags += ['--gbsa', calc.solvent_keyword]

        if calc.distance_constraints or calc.cartesian_constraints or calc.point_charges:
            force_constant = 20

            xcontrol_filename = 'xcontrol_' + calc.name
            with open(xcontrol_filename, 'w') as xcontrol_file:
                if calc.distance_constraints:
                    for atom_ids in calc.distance_constraints.keys():  # xtb counts from 1 so increment atom ids by 1
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

                if calc.point_charges:
                    print(f'$embedding\ninput={calc.name}_xtb.pc\ninput=orca\n$end', file=xcontrol_file)

            calc.flags += ['--input', xcontrol_filename]
            calc.additional_input_files.append(xcontrol_filename)

        if calc.point_charges:
            with open(f'{calc.name}_xtb.pc', 'w') as pc_file:
                print(len(calc.point_charges), file=pc_file)
                for charge, x, y, z in calc.point_charges:
                    print(f'{charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f} 99', file=pc_file)
            calc.additional_input_files.append(f'{calc.name}_xtb.pc')

        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(calc.rev_output_file_lines):
            if 'ERROR' in line:
                return False
            if n_line > 20:
                # With xtb we will search for there being no '#ERROR!' in the last few lines
                return True

    def get_energy(self, calc):
        for line in calc.rev_output_file_lines:
            if 'total E' in line:
                return float(line.split()[-1])
            if 'TOTAL ENERGY' in line:
                return float(line.split()[-3])

    def optimisation_converged(self, calc):

        for line in calc.rev_output_file_lines:
            if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        raise NotImplementedError

    def get_imag_freqs(self, calc):
        raise NotImplementedError

    def get_normal_mode_displacements(self, calc, mode_number):
        raise NotImplementedError

    def _get_final_atoms_6_2_3(self, calc):
        """
        e.g.

        ================
         final structure:
        ================
        5
         xtb: 6.2.3 (830e466)
        Cl        1.62694523673790    0.09780349799138   -0.02455489507427
        C        -0.15839164427314   -0.00942638308615    0.00237760557913
        H        -0.46867957388620   -0.59222865914178   -0.85786049981721
        H        -0.44751262498645   -0.49575975568264    0.92748366742968
        H        -0.55236139359212    0.99971129991918   -0.04744587811734
        """
        atoms = []

        for i, line in enumerate(calc.output_file_lines):
            if 'final structure' in line:
                n_atoms = int(calc.output_file_lines[i+2].split()[0])

                for xyz_line in calc.output_file_lines[i+4:i+4+n_atoms]:
                    atom_label, x, y, z = xyz_line.split()
                    atoms.append(Atom(atom_label, x=float(x), y=float(y), z=float(z)))

                break

        return atoms

    def _get_final_atoms_old(self, calc):
        """
        e.g.

        ================
         final structure:
        ================
        $coord
            2.52072290250473   -0.04782551206377   -0.50388676977877      C
                    .                 .                    .              .
        """
        atoms = []
        geom_section = False

        for line in calc.output_file_lines:

            if '$coord' in line:
                geom_section = True

            if '$end' in line and geom_section:
                geom_section = False

            if len(line.split()) == 4 and geom_section:
                x, y, z, atom_label = line.split()
                atoms.append(Atom(atom_label, x=float(x) * Constants.a02ang, y=float(y) * Constants.a02ang,
                                  z=float(z) * Constants.a02ang))
        return atoms

    def get_final_atoms(self, calc):
        atoms = []

        for i, line in enumerate(calc.output_file_lines):

            # XTB 6.2.x have a slightly different way of printing the atoms, helpfully
            if 'xtb version' in line and len(line.split()) >= 4:
                if line.split()[3] == '6.2.3':
                    atoms = self._get_final_atoms_6_2_3(calc)
                    break

                elif line.split()[3] == '6.2.2' or '6.1' in line.split()[3]:
                    atoms = self._get_final_atoms_old(calc)
                    break

            # Version is not recognised if we're 50 lines into the output file - try and use the old version
            if i > 50:
                atoms = self._get_final_atoms_old(calc)
                break

        if len(atoms) == 0:
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self, calc):
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

    def get_gradients(self, calc):
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

    def __init__(self):
        super().__init__(name='xtb', path=Config.XTB.path, keywords=Config.XTB.keywords)


xtb = XTB()
