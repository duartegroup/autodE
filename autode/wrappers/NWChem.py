import numpy as np
from autode.wrappers.base import ElectronicStructureMethod
from autode.atoms import Atom
from autode.config import Config
from autode.exceptions import UnsuppportedCalculationInput
from autode.log import logger


class NWChem(ElectronicStructureMethod):

    # TODO impliment partial hessian

    def generate_input(self, calc):
        calc.input_filename = calc.name + '_nwchem.nw'
        calc.output_filename = calc.name + '_nwchem.out'
        keywords = calc.keywords_list.copy()

        new_keywords = []
        scf_block = False
        for keyword in keywords:
            if 'opt' in keyword.lower() and calc.molecule.n_atoms == 1:
                logger.warning('Cannot do an optimisation for a single atom')
                old_key = keyword.split()
                new_keyword = ' '
                for word in old_key:
                    if 'opt' in word:
                        new_keyword += 'energy '
                    else:
                        new_keyword += word + ' '
                new_keywords.append(new_keyword)
            elif keyword.lower().startswith('dft'):
                lines = keyword.split('\n')
                lines.insert(1, f'  mult {calc.molecule.mult}')
                new_keyword = '\n'.join(lines)
                new_keywords.append(new_keyword)
            elif keyword.lower().startswith('scf'):
                if calc.solvent_keyword:
                    logger.critical('nwchem only supports solvent for DFT calculations')
                    raise UnsuppportedCalculationInput

                scf_block = True
                lines = keyword.split('\n')
                lines.insert(1, f'  nopen {calc.molecule.mult - 1}')
                new_keyword = '\n'.join(lines)
                new_keywords.append(new_keyword)
            elif any(string in keyword.lower() for string in ['ccsd', 'mp2']) and not scf_block:
                if calc.solvent_keyword:
                    logger.critical('nwchem only supports solvent for DFT calculations')
                    raise UnsuppportedCalculationInput
                new_keywords.append(f'scf\n  nopen {calc.mult - 1}\nend')
                new_keywords.append(keyword)
            else:
                new_keywords.append(keyword)
        keywords = new_keywords

        with open(calc.input_filename, 'w') as inp_file:

            print(f'start {calc.name}_nwchem\necho', file=inp_file)

            if calc.solvent_keyword:
                print(f'cosmo\n do_cosmo_smd true\n solvent {calc.solvent_keyword}\nend', file=inp_file)

            print('geometry', end=' ', file=inp_file)
            if calc.distance_constraints or calc.cartesian_constraints:
                print('noautoz', file=inp_file)
            else:
                print('', file=inp_file)
            for atom in calc.molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}', file=inp_file)
            if hasattr(calc.molecule, 'qm_solvent_atoms') and calc.molecule.qm_solvent_atoms is not None:
                for atom in calc.molecule.qm_solvent_atoms:
                    x, y, z = atom.coord
                    print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}', file=inp_file)
            if calc.bond_ids_to_add or calc.distance_constraints:
                print('  zcoord', file=inp_file)
                if calc.bond_ids_to_add:
                    try:
                        [print('    bond', bond_ids[0] + 1, bond_ids[1] + 1, file=inp_file) for bond_ids in calc.bond_ids_to_add]
                    except (IndexError, TypeError):
                        logger.error('Could not add scanned bond')
                if calc.distance_constraints:
                    [print('    bond', atom_ids[0] + 1, atom_ids[1] + 1, file=inp_file)
                     for atom_ids in calc.distance_constraints.keys()]
                print('  end', file=inp_file)
            print('end', file=inp_file)

            print(f'charge {calc.molecule.charge}', file=inp_file)

            if calc.distance_constraints or calc.cartesian_constraints:
                force_constant = 10
                if calc.constraints_already_met:
                    force_constant += 90
                print('constraints', file=inp_file)
                if calc.distance_constraints:
                    for atom_ids in calc.distance_constraints.keys():  # nwchem counts from 1 so increment atom ids by 1
                        print(f'  spring bond {atom_ids[0] + 1} {atom_ids[1] + 1} {force_constant} {np.round(calc.distance_constraints[atom_ids], 3)}' + str(
                            atom_ids[0] + 1), file=inp_file)

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
                                range_string = str(rang[0]) + ':' + str(rang[-1])
                                list_of_ranges.append(range_string)
                    print('  fix atom', end=' ', file=inp_file)
                    print(*list_of_ranges, sep=' ', file=inp_file)
                print('end', file=inp_file)

            if hasattr(calc.molecule, 'mm_solvent_atoms') and calc.molecule.mm_solvent_atoms is not None:
                print('bq')
                for i, atom in enumerate(calc.molecule.mm_solvent_atoms):
                    charge = calc.molecule.solvent_mol.graph.nodes[i % calc.molecule.solvent_mol.n_atoms]['charge']
                    x, y, z = atom.coord
                    print(f'{x:^12.8f} {y:^12.8f} {z:^12.8f} {charge:^12.8f}', file=inp_file)
                print('end')

            print(f'memory {Config.max_core} mb', file=inp_file)

            print('property\n  mulliken\nend', file=inp_file)

            print(*keywords, sep='\n', file=inp_file)

        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(calc.rev_output_file_lines):
            if any(substring in line for substring in['CITATION',
                                                      'Failed to converge in maximum number of steps or available time']):
                logger.info('nwchem terminated normally')
                return True
            if n_line > 500:
                return False

    def get_energy(self, calc):

        for line in calc.rev_output_file_lines:
            if any(string in line for string in ['Total DFT energy', 'Total SCF energy']):
                return float(line.split()[4])
            if any(string in line for string in ['Total CCSD energy', 'Total CCSD(T) energy', 'Total SCS-MP2 energy', 'Total MP2 energy', 'Total RI-MP2 energy']):
                return float(line.split()[3])

    def optimisation_converged(self, calc):

        for line in calc.rev_output_file_lines:
            if 'Optimization converged' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):

        for j, line in enumerate(calc.rev_output_file_lines):
            if '@' in line:
                if 'ok' in calc.rev_output_file_lines[j-1]:
                    return True

        return False

    def get_imag_freqs(self, calc):

        imag_freqs = []
        normal_mode_section = False

        for line in calc.output_file_lines:
            if 'Projected Frequencies' in line:
                normal_mode_section = True
                imag_freqs = []

            if '------------------------------' in line:
                normal_mode_section = False

            if normal_mode_section and 'P.Frequency' in line:
                freqs = [float(line.split()[i])
                         for i in range(1, len(line.split()))]
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
            if 'Projected Frequencies' in line:
                normal_mode_section = True
                displacements = []

            if '------------------------------' in line:
                normal_mode_section = False

            if normal_mode_section:
                if len(line.split()) == 6:
                    mode_numbers = [int(val) for val in line.split()]
                    if mode_number in mode_numbers:
                        col = [i for i in range(
                            len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 1
                        displacements = [float(disp_line.split()[
                            col]) for disp_line in calc.output_file_lines[j + 4:j + 3 * calc.n_atoms + 4]]

        displacements_xyz = [displacements[i:i + 3]
                             for i in range(0, len(displacements), 3)]
        if len(displacements_xyz) != calc.molecule.n_atoms:
            logger.error(
                'Something went wrong getting the displacements n != n_atoms')
            return None

        return np.array(displacements_xyz)

    def get_final_atoms(self, calc):

        xyzs_section = False
        atoms = []

        for line in calc.output_file_lines:
            if 'Output coordinates in angstroms' in line:
                xyzs_section = True
                atoms = []

            if 'Atomic Mass' in line:
                xyzs_section = False

            if xyzs_section and len(line.split()) == 6:
                if line.split()[0].isdigit():
                    _, atom_label, _, x, y, z = line.split()
                    atoms.append(Atom(atom_label, x=float(x), y=float(y), z=float(z)))

        return atoms

    def get_atomic_charges(self, calc):

        charges_section = False
        charges = []
        for line in calc.rev_output_file_lines:
            if charges_section and len(line.split()) == 4:
                charges.append(float(line.split()[2]) - float(line.split()[3]))
            if '----- Bond indices -----' in line:
                charges_section = True
            if 'gross population on atoms' in line:
                charges_section = False
                return list(reversed(charges))

    def get_gradients(self, calc):

        gradients_section = False
        gradients = []
        for line in calc.rev_output_file_lines:
            if 'DFT ENERGY GRADIENTS' in line:
                gradients_section = True
            if '----------------------------------------' in line:
                gradients_section = False

            if gradients_section and len(line.split()) == 8:
                _, _, _, _, _, x, y, z = line.split()
                gradients.append([float(x), float(y), float(z)])

        return gradients

    def __init__(self):
        super().__init__('nwchem', path=Config.NWChem.path, keywords=Config.NWChem.keywords, mpirun=True)


nwchem = NWChem()
