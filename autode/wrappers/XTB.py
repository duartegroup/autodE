import numpy as np
import os
from autode.wrappers.base import ElectronicStructureMethod
from autode.utils import run_external
from autode.wrappers.keywords import OptKeywords, GradientKeywords
from autode.atoms import Atom
from autode.config import Config
from autode.constants import Constants
from autode.exceptions import AtomsNotFound
from autode.utils import work_in_tmp_dir
from autode.log import logger


def print_distance_constraints(inp_file, molecule, force_constant=20):
    """Add distance constraints to the input file"""

    if molecule.constraints.distance is None:
        return

    for (i, j), dist in molecule.constraints.distance.items():
        # XTB counts from 1 so increment atom ids by 1
        print(f'$constrain\n'
              f'force constant={force_constant}\n'
              f'distance:{i+1}, {j+1}, {dist:.4f}\n$',
              file=inp_file)
    return


def print_cartesian_constraints(inp_file, molecule, force_constant=20):
    """Add cartesian constraints to an xtb input file"""

    if molecule.constraints.cartesian is None:
        return

    constrained_atom_idxs = [i + 1 for i in molecule.constraints.cartesian]
    list_of_ranges, used_atoms = [], []

    for i in constrained_atom_idxs:
        atom_range = []
        if i not in used_atoms:
            while i in constrained_atom_idxs:
                used_atoms.append(i)
                atom_range.append(i)
                i += 1
            if len(atom_range) in (1, 2):
                list_of_ranges += str(atom_range)
            else:
                list_of_ranges.append(f'{atom_range[0]}-{atom_range[-1]}')

    print(f'$constrain\n'
          f'force constant={force_constant}\n'
          f'atoms: {",".join(list_of_ranges)}\n'
          f'$', file=inp_file)
    return


def print_point_charge_file(calc):
    """Generate a point charge file"""

    if calc.input.point_charges is None:
        return

    with open(f'{calc.name}_xtb.pc', 'w') as pc_file:
        print(len(calc.input.point_charges), file=pc_file)

        for point_charge in calc.input.point_charges:
            x, y, z = point_charge.coord
            charge = point_charge.charge
            print(f'{charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f}', file=pc_file)

    calc.input.additional_filenames.append(f'{calc.name}_xtb.pc')
    return


def print_xcontrol_file(calc, molecule):
    """Print an XTB input file with constraints and point charges"""

    xcontrol_filename = f'xcontrol_{calc.name}'
    with open(xcontrol_filename, 'w') as xcontrol_file:

        print_distance_constraints(xcontrol_file, molecule)
        print_cartesian_constraints(xcontrol_file, molecule)

        if calc.input.point_charges is not None:
            print_point_charge_file(calc)
            print(f'$embedding\n'
                  f'input={calc.name}_xtb.pc\n'
                  f'input=orca\n'
                  f'$end', file=xcontrol_file)

    calc.input.additional_filenames.append(xcontrol_filename)
    return


class XTB(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):

        calc.molecule.print_xyz_file(filename=calc.input.filename)

        if molecule.constraints.any() or calc.input.point_charges:
            print_xcontrol_file(calc, molecule)

        return None

    def get_input_filename(self, calc):
        return f'{calc.name}.xyz'

    def get_output_filename(self, calc):
        return f'{calc.name}.out'

    def execute(self, calc):
        """Execute an XTB calculation using the runtime flags"""
        # XTB calculation keywords must be a class

        flags = ['--chrg', str(calc.molecule.charge)]

        if isinstance(calc.input.keywords, OptKeywords):
            flags.append('--opt')

        if isinstance(calc.input.keywords, GradientKeywords):
            flags.append('--grad')

        if calc.input.solvent is not None:
            flags += ['--gbsa', calc.input.solvent]

        if len(calc.input.additional_filenames) > 0:
            # XTB allows for an additional xcontrol file, which should be the
            # last file in the list
            flags += ['--input', calc.input.additional_filenames[-1]]

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.xyz', '.out', '.pc', '.grad', 'gradient'))
        def execute_xtb():
            logger.info(f'Setting the number of OMP threads to {calc.n_cores}')
            os.environ['OMP_NUM_THREADS'] = str(calc.n_cores)

            run_external(params=[calc.method.path, calc.input.filename]+flags,
                         output_filename=calc.output.filename)

        execute_xtb()
        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if 'ERROR' in line:
                return False
            if n_line > 20:
                # With xtb we will search for there being no '#ERROR!' in the
                # last few lines
                return True

        return False

    def get_energy(self, calc):
        for line in reversed(calc.output.file_lines):
            if 'total E' in line:
                return float(line.split()[-1])
            if 'TOTAL ENERGY' in line:
                return float(line.split()[-3])

    def get_enthalpy(self, calc):
        raise NotImplementedError

    def get_free_energy(self, calc):
        raise NotImplementedError

    def optimisation_converged(self, calc):

        for line in reversed(calc.output.file_lines):
            if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        raise NotImplementedError

    def get_imaginary_freqs(self, calc):
        raise NotImplementedError

    def get_normal_mode_displacements(self, calc, mode_number):
        raise NotImplementedError

    def _get_final_atoms_6_2_above(self, calc):
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

        for i, line in enumerate(calc.output.file_lines):
            if 'final structure' in line:
                n_atoms = int(calc.output.file_lines[i+2].split()[0])

                for xyz_line in calc.output.file_lines[i+4:i+4+n_atoms]:
                    atom_label, x, y, z = xyz_line.split()
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))

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

        for line in calc.output.file_lines:

            if '$coord' in line:
                geom_section = True

            if '$end' in line and geom_section:
                geom_section = False

            if len(line.split()) == 4 and geom_section:
                x, y, z, atom_label = line.split()

                atom = Atom(atom_label,
                            x=float(x) * Constants.a02ang,
                            y=float(y) * Constants.a02ang,
                            z=float(z) * Constants.a02ang)

                atoms.append(atom)

        return atoms

    def get_final_atoms(self, calc):
        atoms = []

        for i, line in enumerate(calc.output.file_lines):

            # XTB 6.2.x have a slightly different way of printing the atoms
            if 'xtb version' in line and len(line.split()) >= 4:
                if line.split()[3] == '6.2.3' or '6.3' in line.split()[3]:
                    atoms = self._get_final_atoms_6_2_above(calc)
                    break

                elif line.split()[3] == '6.2.2' or '6.1' in line.split()[3]:
                    atoms = self._get_final_atoms_old(calc)
                    break

            # Version is not recognised if we're 50 lines into the output file
            # - try and use the old version
            if i > 50:
                atoms = self._get_final_atoms_old(calc)
                break

        if len(atoms) == 0:
            raise AtomsNotFound

        return atoms

    def get_atomic_charges(self, calc):
        charges_sect = False
        charges = []
        for line in calc.output.file_lines:
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
                    gradients.append(np.array([float(x), float(y), float(z)]))

        elif os.path.exists('gradient'):
            with open('gradient', 'r') as grad_file:
                for i, line in enumerate(grad_file):
                    if i > 1 and len(line.split()) == 3:
                        x, y, z = line.split()
                        vec = [float(x.replace('D', 'E')),
                               float(y.replace('D', 'E')),
                               float(z.replace('D', 'E'))]

                        gradients.append(np.array(vec))

            with open(f'{calc.name}_xtb.grad', 'w') as new_grad_file:
                [print('{:^12.8f} {:^12.8f} {:^12.8f}'.format(*line),
                       file=new_grad_file) for line in gradients]
            os.remove('gradient')

        # Convert from Ha a0^-1 to Ha A-1
        gradients = [grad / Constants.a02ang for grad in gradients]
        return np.array(gradients)

    def __init__(self):
        super().__init__(name='xtb', path=Config.XTB.path,
                         keywords_set=Config.XTB.keywords,
                         implicit_solvation_type=Config.XTB.implicit_solvation_type)


xtb = XTB()
