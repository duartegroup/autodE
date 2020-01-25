from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods
import numpy as np
import os

smd_solvents = ['1,1,1-TRICHLOROETHANE', 'CYCLOPENTANE', '1,1,2-TRICHLOROETHANE', 'CYCLOPENTANOL',
                '1,2,4-TRIMETHYLBENZENE', 'CYCLOPENTANONE', '1,2-DIBROMOETHANE', '1,2-DICHLOROETHANE ',
                'CIS-DECALIN',
                '1,2-ETHANEDIOL ', 'N-DECANE', '1,4-DIOXANEDIBROMOMETHANE', '1-BROMO-2-METHYLPROPANE',
                'DIBUTYLETHER',
                '1-BROMOOCTANE', 'O-DICHLOROBENZENE', '1-BROMOPENTANE', 'E-1,2-DICHLOROETHENE', '1-BROMOPROPANE ',
                'Z-1,2-DICHLOROETHENE', '1-BUTANOL', 'DICHLOROMETHANE', '1-CHLOROHEXANE', '1-CHLOROPENTANE',
                '1-CHLOROPROPANE', 'DIETHYLAMINE', '1-DECANOL', 'DIIODOMETHANE', '1-FLUOROOCTANE', '1-HEPTANOL',
                'CIS-1,2-DIMETHYLCYCLOHEXANE', 'DECALIN', 'DIETHYL ETHER ', 'DIETHYL SULFIDE', 'DIISOPROPYL ETHER',
                '1-HEXANOL', '1-HEXENE', 'N,N-DIMETHYLACETAMIDE', '1-HEXYNE', 'N,N-DIMETHYLFORMAMIDE DMF',
                '1-IODOBUTANE', 'DIMETHYLSULFOXIDE DMSO', '1-IODOHEXADECANE', 'DIPHENYLETHER', '1-IODOPENTANE',
                'DIPROPYLAMINE', '1-IODOPROPANE', 'N-DODECANE', '1-NITROPROPANE', 'ETHANETHIOL', '1-NONANOL',
                'ETHANOL',
                '1-OCTANOL', '1-PENTANOL', '1-PENTENE', '1-PROPANOL', 'ETHYLBENZENE', '2,2,2-TRIFLUOROETHANOL',
                'LUOROBENZENE', '2,2,4-TRIMETHYLPENTANE', 'FORMAMIDE', '2,4-DIMETHYLPENTANE',
                '2,4-DIMETHYLPYRIDINE',
                'N-HEPTANE', '2,6-DIMETHYLPYRIDINE', 'N-HEXADECANE', '2-BROMOPROPANE', 'N-HEXANE',
                'DIMETHYL DISULFIDE',
                'ETHYL ETHANOATE ', 'ETHYL METHANOATE ', 'ETHYL PHENYL ETHER', 'FORMIC ACID', '2-BUTANOL',
                'HEXANOIC ACID', '2-CHLOROBUTANE', '2-HEPTANONE', '2-HEXANONE', '2-METHOXYETHANOL',
                '2-METHYL-1-PROPANOL', '2-METHYL-2-PROPANOL', '2-METHYLPENTANE', '2-METHYLPYRIDINE',
                '2-NITROPROPANE', '2-OCTANONE', '2-PENTANONE', 'IODOBENZENE', 'IODOETHANE', 'IODOMETHANE',
                'ISOPROPYLBENZENE', 'P-ISOPROPYLTOLUENE', 'MESITYLENE', 'METHANOL', 'METHYL BENZOATE',
                'METHYL BUTANOATE', 'METHYL ETHANOATE', 'METHYL METHANOATE', 'METHYL PROPANOATE', 'N-METHYLANILINE',
                'METHYLCYCLOHEXANE', 'N-METHYLFORMAMIDE (E/Z MIXTURE)', 'NITROBENZENE', 'PhNO2', 'NITROETHANE',
                'NITROMETHANE', 'MeNO2 ', 'O-NITROTOLUENE', 'N-NONANE', 'N-OCTANE', 'N-PENTADECANE', 'PENTANAL',
                'N-PENTANE', 'PENTANOIC ACID', 'PENTYL ETHANOATE', 'PENTYLAMINE', 'PERFLUOROBENZENE', 'PROPANAL',
                'PROPANOIC ACID', 'PROPANONITRILE', 'PROPYL ETHANOATE', 'PROPYLAMINE', 'PYRIDINE',
                'TETRACHLOROETHENE',
                'TETRAHYDROFURAN', 'THF', 'TETRAHYDROTHIOPHENE-S,S-DIOXIDE', 'TETRALIN', 'THIOPHENE', 'THIOPHENOL',
                'TOLUENE', 'TRANS-DECALIN', 'TRIBUTYLPHOSPHATE', 'TRICHLOROETHENE', 'TRIETHYLAMINE', 'N-UNDECANE',
                'WATER', 'XYLENE (MIXTURE)', 'M-XYLENE', 'O-XYLENE', 'P-XYLENE', '2-PROPANOL', '2-PROPEN-1-OL',
                'E-2-PENTENE', '3-METHYLPYRIDINE', '3-PENTANONE', '4-HEPTANONE', '4-METHYL-2-PENTANONE',
                '4-METHYLPYRIDINE', '5-NONANONE', 'ACETIC ACID', 'ACETONE', 'ACETONITRILE MeCN', 'ACETOPHENONE',
                'ANILINE', 'ANISOLE', 'BENZALDEHYDE', 'BENZENE', 'BENZONITRILE', 'BENZYL ALCOHOL', 'BROMOBENZENE',
                'BROMOETHANE', 'BROMOFORM', 'BUTANAL', 'BUTANOIC ACID', 'BUTANONE', 'BUTANONITRILE',
                'BUTYL ETHANOATE',
                'BUTYLAMINE', 'N-BUTYLBENZENE', 'SEC-BUTYLBENZENE', 'TERT-BUTYLBENZENE', 'CARBON DISULFIDE',
                'CARBON TETRACHLORIDE', 'CHLOROBENZENE', 'CHLOROFORM', 'A-CHLOROTOLUENE', 'O-CHLOROTOLUENE',
                'M-CRESOL',
                'O-CRESOL', 'CYCLOHEXANE', 'CYCLOHEXANONE']

ORCA = ElectronicStructureMethod(name='orca', path=Config.ORCA.path,
                                 aval_solvents=[solv.lower()
                                                for solv in smd_solvents],
                                 scan_keywords=Config.ORCA.scan_keywords,
                                 conf_opt_keywords=Config.ORCA.conf_opt_keywords,
                                 opt_keywords=Config.ORCA.opt_keywords,
                                 opt_ts_keywords=Config.ORCA.opt_ts_keywords,
                                 hess_keywords=Config.ORCA.hess_keywords,
                                 opt_ts_block=Config.ORCA.opt_ts_block,
                                 sp_keywords=Config.ORCA.sp_keywords)

ORCA.__name__ = 'ORCA'


def generate_input(calc):
    calc.input_filename = calc.name + '_orca.inp'
    calc.output_filename = calc.name + '_orca.out'
    keywords = calc.keywords.copy()

    if calc.n_atoms == 1:
        for keyword in keywords:
            if 'opt' in keyword.lower():
                logger.warning('Cannot do an optimisation for a single atom')
                keywords.remove(keyword)

    with open(calc.input_filename, 'w') as inp_file:
        print('!', *keywords, file=inp_file)

        if calc.solvent:
            print('%cpcm\n smd true\n SMDsolvent \"' +
                  calc.solvent + '\"\n end', file=inp_file)

        if calc.optts_block:
            print(calc.optts_block, file=inp_file)
            if calc.core_atoms and calc.n_atoms > 25:
                core_atoms_str = ' '.join(map(str, calc.core_atoms))
                print(f'Hybrid_Hess [{core_atoms_str}] end', file=inp_file)
            print('end', file=inp_file)

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

        if calc.n_atoms < 33:
            print('%geom MaxIter 100 end', file=inp_file)

        if calc.n_cores > 1:
            print('%pal nprocs ' + str(calc.n_cores) + '\nend', file=inp_file)
        print('%output \nxyzfile=True \nend ', file=inp_file)
        print('%scf \nmaxiter 250 \nend', file=inp_file)
        print('% maxcore', calc.max_core_mb, file=inp_file)
        print('*xyz', calc.charge, calc.mult, file=inp_file)
        [print('{:<3} {:^12.8f} {:^12.8f} {:^12.8f}'.format(*line), file=inp_file) for line in calc.xyzs]
        print('*', file=inp_file)

    return None


def calculation_terminated_normally(calc):

    for n_line, line in enumerate(calc.rev_output_file_lines):
        if any(substring in line for substring in['ORCA TERMINATED NORMALLY', 'The optimization did not converge', 'HUGE, UNRELIABLE STEP WAS ABOUT TO BE TAKEN']):
            logger.info('ORCA terminated normally')
            return True
        if n_line > 30:
            # The above lines are pretty close to the end of the file – there's no point parsing it all
            return False


def get_energy(calc):
    for line in calc.rev_output_file_lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            return float(line.split()[4])


def optimisation_converged(calc):

    for line in calc.rev_output_file_lines:
        if 'THE OPTIMIZATION HAS CONVERGED' in line:
            return True

    return False


def optimisation_nearly_converged(calc):
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


def get_imag_freqs(calc):
    imag_freqs = None

    for i, line in enumerate(calc.output_file_lines):
        if 'VIBRATIONAL FREQUENCIES' in line:
            freq_lines = calc.output_file_lines[i + 5:i + 3 * calc.n_atoms + 5]
            freqs = [float(l.split()[1]) for l in freq_lines]
            imag_freqs = [freq for freq in freqs if freq < 0]

    logger.info(f'Found imaginary freqs {imag_freqs}')
    return imag_freqs


def get_normal_mode_displacements(calc, mode_number):
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
                                     calc.output_file_lines[j + 1:j + 3 * calc.n_atoms + 1]]

    displacements_xyz = [displacements[i:i + 3]
                         for i in range(0, len(displacements), 3)]
    if len(displacements_xyz) != calc.n_atoms:
        logger.error('Something went wrong getting the displacements n != n_atoms')
        return None

    return displacements_xyz


def get_final_xyzs(calc):

    xyzs = []
    if calc.output_filename:
        xyz_file_name = calc.output_filename[:-4] + '.xyz'
        if os.path.exists(xyz_file_name):
            with open(xyz_file_name, 'r') as file:
                for line_no, line in enumerate(file):
                    if line_no > 1:
                        atom_label, x, y, z = line.split()
                        xyzs.append([atom_label, float(x), float(y), float(z)])

    return xyzs


# Bind all the required functions to the class definition
[setattr(ORCA, method, globals()[method]) for method in req_methods]
