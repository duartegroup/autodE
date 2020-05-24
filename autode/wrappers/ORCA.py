import numpy as np
import os
from autode.wrappers.base import ElectronicStructureMethod
from autode.atoms import Atom
from autode.config import Config
from autode.exceptions import UnsuppportedCalculationInput
from autode.exceptions import NoCalculationOutput
from autode.exceptions import NoNormalModesFound
from autode.log import logger

vdw_gaussian_solvent_dict = {'water': 'Water', 'acetone': 'Acetone', 'acetonitrile': 'Acetonitrile', 'benzene': 'Benzene',
                             'carbon tetrachloride': 'CCl4', 'dichloromethane': 'CH2Cl2', 'chloroform': 'Chloroform', 'cyclohexane': 'Cyclohexane',
                             'n,n-dimethylformamide': 'DMF', 'dimethylsulfoxide': 'DMSO', 'ethanol': 'Ethanol', 'n-hexane': 'Hexane',
                             'methanol': 'Methanol', '1-octanol': 'Octanol', 'pyridine': 'Pyridine', 'tetrahydrofuran': 'THF', 'toluene': 'Toluene'}


class ORCA(ElectronicStructureMethod):

    def generate_input(self, calc):

        calc.input_filename = calc.name + '_orca.inp'
        calc.output_filename = calc.name + '_orca.out'
        keywords = calc.keywords_list.copy()

        use_vdw_gaus_solvent = True if Config.ORCA.solvation_type.lower() == 'cpcm' else False

        if any('freq' in keyword.lower() or 'optts' in keyword.lower() for keyword in keywords) and use_vdw_gaus_solvent:
            logger.error('Cannot do analytical frequencies with gaussian charge scheme - switching off')
            use_vdw_gaus_solvent = False

        qmmm_freq = False

        for keyword in keywords:
            if 'opt' in keyword.lower():
                if calc.n_atoms == 1:
                    logger.warning('Cannot do an optimisation for a single atom')
                    keywords.remove(keyword)    # ORCA defaults to a single point calculation

            if keyword.lower() == 'freq' or keyword.lower() == 'optts':
                if hasattr(calc.molecule, 'qm_solvent_atoms') and calc.molecule.qm_solvent_atoms:
                    logger.warning('Cannot do analytical freqencies with point charges')

                    keywords.remove(keyword)
                    keywords.append('NumFreq')
                    qmmm_freq = True

        if calc.solvent_keyword is not None:
            if Config.ORCA.solvation_type.lower() not in ['smd', 'cpcm']:
                raise UnsuppportedCalculationInput

            if Config.ORCA.solvation_type.lower() == 'smd':
                keywords.append('CPCM')

            if Config.ORCA.solvation_type.lower() == 'cpcm':
                if calc.solvent_keyword not in vdw_gaussian_solvent_dict.keys():
                    raise UnsuppportedCalculationInput(message=f'CPCM solvent with gaussian charge '
                                                               f'not avalible for {calc.solvent_keyword}')

                keywords.append(f'CPCM({vdw_gaussian_solvent_dict[calc.solvent_keyword]})')

        with open(calc.input_filename, 'w') as inp_file:
            print('!', *keywords, file=inp_file)

            if calc.solvent_keyword is not None:

                if Config.ORCA.solvation_type.lower() == 'smd':
                    print(f'%cpcm\nsmd true\nSMDsolvent \"{calc.solvent_keyword}\"\nend', file=inp_file)

                if use_vdw_gaus_solvent:
                    print('%cpcm\n surfacetype vdw_gaussian\nend', file=inp_file)

            max_iter_done = False
            if calc.other_input_block:
                if 'maxiter' in calc.other_input_block.lower():
                    max_iter_done = True
                print(calc.other_input_block, file=inp_file)

            if calc.bond_ids_to_add:
                try:
                    [print('%geom\nmodify_internal\n{ B', bond_ids[0], bond_ids[1], 'A } end\nend', file=inp_file)
                     for bond_ids in calc.bond_ids_to_add]
                except (IndexError, TypeError):
                    logger.error('Could not add scanned bond')

            if calc.distance_constraints:
                print('%geom Constraints', file=inp_file)
                for bond_ids in calc.distance_constraints.keys():
                    print('{ B', bond_ids[0], bond_ids[1], calc.distance_constraints[bond_ids], 'C }',
                          file=inp_file)
                print('    end\nend', file=inp_file)

            if calc.cartesian_constraints:
                print('%geom Constraints', file=inp_file)
                [print('{ C', atom_id, 'C }', file=inp_file)
                 for atom_id in calc.cartesian_constraints]
                print('    end\nend', file=inp_file)

            if calc.n_atoms < 33 and not max_iter_done:
                print('%geom MaxIter 100 end', file=inp_file)

            if qmmm_freq:
                print('%freq\nPartial_Hess {', file=inp_file, end='')
                solvent_atoms = [i + calc.molecule.n_atoms for i in range(len(calc.molecule.qm_solvent_atoms))]
                print(*solvent_atoms, file=inp_file, end='')
                print('} end\nend', file=inp_file)

            if calc.point_charges is not None:
                with open(f'{calc.name}_orca.pc', 'w') as pc_file:
                    print(len(calc.point_charges), file=pc_file)
                    for point_charge in calc.point_charges:
                        x, y, z = point_charge.coord
                        print(f'{point_charge.charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f}', file=pc_file)
                    calc.additional_input_files.append(f'{calc.name}_orca.pc')
                print(f'% pointcharges "{calc.name}_orca.pc"', file=inp_file)

            if calc.n_cores > 1:
                print('%pal nprocs ' + str(calc.n_cores) + '\nend', file=inp_file)

            print('%output \nxyzfile=True \nend ',
                  '%scf \nmaxiter 250 \nend',
                  '%output\nPrint[P_Hirshfeld] = 1\nend',
                  '% maxcore', calc.max_core_mb, sep='\n', file=inp_file)

            print('*xyz', calc.molecule.charge, calc.molecule.mult, file=inp_file)
            for atom in calc.molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}', file=inp_file)
            print('*', file=inp_file)

        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(calc.rev_output_file_lines):
            if any(substring in line for substring in['ORCA TERMINATED NORMALLY', 'The optimization did not converge']):
                logger.info('orca terminated normally')
                return True
            if n_line > 30:
                # The above lines are pretty close to the end of the file – there's no point parsing it all
                return False

    def get_energy(self, calc):
        for line in calc.rev_output_file_lines:
            if 'FINAL SINGLE POINT ENERGY' in line:
                return float(line.split()[4])

    def get_enthalpy(self, calc):
        """Get the enthalpy (H) from an ORCA calculation output"""

        for line in calc.rev_output_file_lines:
            if 'Total Enthalpy' in line:

                try:
                    return float(line.split()[-2])

                except ValueError:
                    break

        logger.error('Could not get the free energy from the calculation. Was a frequency requested?')
        return None

    def get_free_energy(self, calc):
        """Get the Gibbs free energy (G) from an ORCA calculation output"""

        for line in calc.rev_output_file_lines:
            if 'Final Gibbs free enthalpy' in line:

                try:
                    return float(line.split()[-2])

                except ValueError:
                    break

        logger.error('Could not get the free energy from the calculation. Was a frequency requested?')
        return None

    def optimisation_converged(self, calc):

        for line in calc.rev_output_file_lines:
            if 'THE OPTIMIZATION HAS CONVERGED' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        geom_conv_block = False

        for line in calc.rev_output_file_lines:
            if geom_conv_block and 'Geometry convergence' in line:
                geom_conv_block = False
            if 'The optimization has not yet converged' in line:
                geom_conv_block = True
            if geom_conv_block and len(line.split()) == 5:
                if line.split()[-1] == 'YES':
                    return True
        return False

    def get_imag_freqs(self, calc):
        imag_freqs = []

        for i, line in enumerate(calc.output_file_lines):
            if 'VIBRATIONAL FREQUENCIES' in line:
                freq_lines = calc.output_file_lines[i + 5:i + 3 * calc.molecule.n_atoms + 5]
                freqs = [float(l.split()[1]) for l in freq_lines]
                imag_freqs = [freq for freq in freqs if freq < 0]

        logger.info(f'Found imaginary freqs {imag_freqs}')
        return imag_freqs

    def get_normal_mode_displacements(self, calc, mode_number):
        normal_mode_section, values_sec, displacements, col = False, False, [], None

        for j, line in enumerate(calc.output_file_lines):
            if 'NORMAL MODES' in line:
                normal_mode_section, values_sec, displacements, col = True, False, [], None

            if 'IR SPECTRUM' in line:
                normal_mode_section, values_sec = False, False

            if normal_mode_section and len(line.split()) > 1:
                if line.split()[0].startswith('0'):
                    values_sec = True

            if values_sec:
                if '.' not in line and len(line.split()) > 1:
                    mode_numbers = [int(val) for val in line.split()]
                    if mode_number in mode_numbers:
                        col = [i for i in range(len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 1
                        displacements = [float(disp_line.split()[col]) for disp_line in
                                         calc.output_file_lines[j + 1:j + 3 * calc.molecule.n_atoms + 1]]

        displacements_xyz = [displacements[i:i + 3] for i in range(0, len(displacements), 3)]

        if len(displacements_xyz) != calc.molecule.n_atoms:
            logger.error('Something went wrong getting the displacements n != n_atoms')
            raise NoNormalModesFound

        return np.array(displacements_xyz)

    def get_final_atoms(self, calc):

        atoms = []
        xyz_file_name = calc.output_filename.replace('.out', '.xyz')

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

        for i, line in enumerate(calc.output_file_lines):
            if 'HIRSHFELD ANALYSIS' in line:
                charges = []
                for charge_line in calc.output_file_lines[i+7:i+7+calc.n_atoms]:
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

        for i, line in enumerate(calc.output_file_lines):
            if 'CARTESIAN GRADIENT' in line:
                gradients = []
                j = i + 3
                for grad_line in calc.output_file_lines[j:j+calc.n_atoms]:
                    dadx, dady, dadz = grad_line.split()[-3:]
                    gradients.append([float(dadx), float(dady), float(dadz)])

        return gradients

    def __init__(self):
        super().__init__(name='orca', path=Config.ORCA.path, keywords=Config.ORCA.keywords)


orca = ORCA()
