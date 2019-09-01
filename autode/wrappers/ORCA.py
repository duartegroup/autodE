from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods
import numpy as np

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

ORCA = ElectronicStructureMethod(name='orca',
                                 path=Config.ORCA.path,
                                 aval_solvents=[solv.lower() for solv in smd_solvents])


def generate_input(calc):
    calc.input_filename = calc.name + '_orca.inp'
    calc.output_filename = calc.name + '_orca.out'

    if len(calc.xyzs) == 1:
        for keyword in calc.keywords:
            if keyword.lower() == 'opt' or keyword.lower() == 'looseopt' or keyword.lower() == 'tightopt':
                logger.warning('Cannot do an optimisation for a single atom')
                calc.keywords.remove(keyword)

    with open(calc.input_filename, 'w') as inp_file:
        print('!', *calc.keywords, file=inp_file)

        if calc.solvent:
            print('%cpcm\n smd true\n SMDsolvent \"' + calc.solvent + '\"\n end', file=inp_file)

        if calc.optts_block:
            print(calc.optts_block, file=inp_file)

        if calc.bond_ids_to_add:
            try:
                [print('%geom\nmodify_internal\n{ B', bond_ids[0], bond_ids[1], 'A } end\nend', file=inp_file)
                 for bond_ids in calc.bond_ids_to_add]
            except IndexError or TypeError:
                logger.error('Could not add scanned bond')

        if calc.scan_ids:
            try:
                print('%geom Scan\n    B', calc.scan_ids[0], calc.scan_ids[1],
                      '= ' + str(np.round(calc.curr_d1, 3)) + ', ' +
                      str(np.round(calc.final_d1, 3)) + ', ' + str(calc.n_steps) + '\n    end\nend',
                      file=inp_file)

                if calc.scan_ids2 is not None:
                    print('%geom Scan\n    B', calc.scan_ids2[0], calc.scan_ids2[1],
                          '= ' + str(np.round(calc.curr_d2, 3)) + ', ' +
                          str(np.round(calc.final_d2, 3)) + ', ' + str(calc.n_steps) + '\n    end\nend',
                          file=inp_file)

            except IndexError:
                logger.error('Could not add scan block')

        if calc.distance_constraints:
            print('%geom Constraints', file=inp_file)
            for bond_ids in calc.distance_constraints.keys():
                print('{ B', bond_ids[0], bond_ids[1], calc.distance_constraints[bond_ids], 'C }',
                      file=inp_file)
            print('    end\nend', file=inp_file)

        if len(calc.xyzs) < 33:
            print('%geom MaxIter 100 end', file=inp_file)

        if calc.n_cores > 1:
            print('%pal nprocs ' + str(calc.n_cores) + '\nend', file=inp_file)
        print('%scf \nmaxiter 250 \nend', file=inp_file)
        print('% maxcore', calc.max_core_mb, file=inp_file)
        print('*xyz', calc.charge, calc.mult, file=inp_file)
        [print('{:<3}{:^12.8f}{:^12.8f}{:^12.8f}'.format(*line), file=inp_file) for line in calc.xyzs]
        print('*', file=inp_file)

    return None


def calculation_terminated_normally(calc):

    for n_line, line in enumerate(calc.rev_output_file_lines):
        if 'ORCA TERMINATED NORMALLY' in line or 'The optimization did not converge' in line:
            logger.info('ORCA terminated normally')
            return True
        if n_line > 20:
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

    logger.info('Found imaginary freqs {}'.format(imag_freqs))
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

        if values_sec and len(line.split()) == 6:
            mode_numbers = [int(val) for val in line.split()]
            if mode_number in mode_numbers:
                col = [i for i in range(len(mode_numbers)) if mode_number == mode_numbers[i]][0] + 1
                displacements = [float(disp_line.split()[col]) for disp_line in
                                 calc.output_file_lines[j + 1:j + 3 * calc.n_atoms + 1]]

    displacements_xyz = [displacements[i:i + 3] for i in range(0, len(displacements), 3)]
    if len(displacements_xyz) != calc.n_atoms:
        logger.error('Something went wrong getting the displacements n != n_atoms')
        return None

    return displacements_xyz


def get_final_xyzs(calc):

    xyzs = []
    xyz_section = False

    for line in calc.rev_output_file_lines:

        if 'CARTESIAN COORDINATES (A.U.)' in line:
            xyz_section = True
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line and xyz_section:
            break

        if xyz_section and len(line.split()) == 4:
            atom_label, x, y, z = line.split()
            xyzs.append([atom_label, float(x), float(y), float(z)])

    return xyzs


def get_scan_values_xyzs_energies(calc):
    logger.info('Getting the xyzs and energies from an ORCA relaxed PES scan')
    scan_2d = True if calc.scan_ids2 is not None else False

    def get_orca_scan_values_xyzs_energies_no_conv(out_lines, scan_2d=False, delta_e_threshold_kcal_mol=1.0):

        logger.info('Getting the xyzs and energies from a non-converged ORCA relaxed PES scan')

        values_xyzs_energies, curr_dist, curr_dist1, curr_dist2, n_atoms = {}, None, None, None, 0
        curr_energy, curr_delta_energy, scan_point_xyzs = 0.0, 0.0, []

        for n_line, line in enumerate(out_lines):
            if 'Number of atoms' in line:
                n_atoms = int(line.split()[-1])

            if 'RELAXED SURFACE SCAN STEP' in line:
                if scan_2d:
                    curr_dist1 = float(out_lines[n_line + 2].split()[-2])
                    curr_dist2 = float(out_lines[n_line + 3].split()[-2])
                else:
                    curr_dist = float(out_lines[n_line + 2].split()[-2])

            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                scan_point_xyzs = []
                for xyz_line in out_lines[n_line + 2:n_line + 1 + n_atoms + 1]:
                    atom_label, x, y, z = xyz_line.split()
                    scan_point_xyzs.append([atom_label, float(x), float(y), float(z)])

            if 'FINAL SINGLE POINT ENERGY' in line:
                curr_delta_energy = np.abs(float(line.split()[4]) - curr_energy)
                curr_energy = float(line.split()[4])

            if 'RELAXED SURFACE SCAN STEP' in line or 'ORCA TERMINATED NORMALLY' in line:
                if scan_2d:
                    # Consider everything converged – perhaps not a great idea
                    if curr_dist1 is not None and curr_dist2 is not None and curr_energy != 0.0:
                        values_xyzs_energies[(curr_dist1, curr_dist2)] = scan_point_xyzs, curr_energy

                else:
                    if curr_dist is not None and curr_energy != 0.0:
                        if Constants.ha2kcalmol * curr_delta_energy < delta_e_threshold_kcal_mol:
                            values_xyzs_energies[curr_dist] = scan_point_xyzs, curr_energy
                        else:
                            logger.warning('Optimisation wasn\'t close to converging on this step')

        return values_xyzs_energies

    values_xyzs_energies = {}
    curr_dist1, curr_dist2, curr_dist = 0, 0, 0
    scan_point_xyzs, scan_point_energy, opt_done, xyz_block = [], 0, False, False

    for n_line, line in enumerate(calc.output_file_lines):
        if 'The optimization did not converge' in line:
            logger.warning('Optimisation did not converge')
            return get_orca_scan_values_xyzs_energies_no_conv(calc.output_file_lines, scan_2d=scan_2d)

        if 'RELAXED SURFACE SCAN STEP' in line:
            scan_point_xyzs, opt_done, xyz_block = [], False, False
            if scan_2d:
                curr_dist1 = float(calc.output_file_lines[n_line + 2].split()[-2])
                curr_dist2 = float(calc.output_file_lines[n_line + 3].split()[-2])
            else:
                curr_dist = float(calc.output_file_lines[n_line + 2].split()[-2])

        if 'THE OPTIMIZATION HAS CONVERGED' in line:
            opt_done = True
        if 'CARTESIAN COORDINATES' in line and opt_done:
            xyz_block = True

        if xyz_block and len(line.split()) == 4:
            atom_label, x, y, z = line.split()
            scan_point_xyzs.append([atom_label, float(x), float(y), float(z)])

        if xyz_block and len(line.split()) == 0:
            xyz_block = False

        if opt_done and len(scan_point_xyzs) > 0:
            if 'FINAL SINGLE POINT ENERGY' in line:
                scan_point_energy = float(line.split()[4])

            if scan_2d:
                values_xyzs_energies[(curr_dist1, curr_dist2)] = scan_point_xyzs, scan_point_energy
            else:
                values_xyzs_energies[curr_dist] = scan_point_xyzs, scan_point_energy

    if len(values_xyzs_energies) == 0:
        logger.error('Could not get any energies or xyzs from ORCA PES scan')
        return None


# Bind all the required functions to the class definition
[setattr(ORCA, method, globals()[method]) for method in req_methods]
