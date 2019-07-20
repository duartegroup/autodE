import os
import numpy as np
from copy import deepcopy
from .constants import Constants
from subprocess import Popen
from .config import Config
from .log import logger


def gen_orca_inp(inp_filename, keywords, xyzs, charge, mult, solvent=None, n_cores=1, add_bond_ids=None, optts=False,
                 scan_ids=None, curr_dist1=1.5, final_dist1=3.5, curr_dist2=1.5, final_dist2=3.5, n_steps=10,
                 scan_ids2=None, distance_constraints=None):

    logger.info('Generating {}'.format(inp_filename))

    keywrds = deepcopy(keywords)

    if 1 < n_cores <= 8:
        keywrds.append('PAL' + str(n_cores))
    if n_cores > 8:
        logger.warning('ORCA calculations are capped at using 8 core. Reducing from {} to 8'.format(n_cores))
        keywrds.append('PAL8')

    if not inp_filename.endswith('.inp'):
        logger.warning('inp_filename doesn\'t end with .inp. Adding it')
        inp_filename += '.inp'

    if xyzs is None or len(xyzs) == 0:
        logger.critical('Have no xyzs. Can\'t generate a valid .inp file')
        exit()

    if len(xyzs) == 1:
        for keyword in keywrds:
            if keyword.lower() == 'opt' or keyword.lower() == 'looseopt':
                logger.warning('Cannot do an optimisation for a single atom')
                keywrds.remove(keyword)
                keywrds.append('SP')
                break

    return write_inp_file(inp_filename, keywrds, xyzs, charge, mult, solvent, add_bond_ids, optts, scan_ids,
                          curr_dist1, final_dist1, curr_dist2, final_dist2, n_steps, scan_ids2, distance_constraints)


def write_inp_file(inp_filename, keywords, xyzs, charge, mult, solvent, add_bond_ids, optts, scan_ids, curr_dist1,
                   final_dist1, curr_dist2, final_dist2, n_steps, scan_ids2, distance_constraints):

    with open(inp_filename, 'w') as inp_file:
        print('!', *keywords, file=inp_file)
        if solvent:
            # TODO check whether solvent is in the list of SMD solvent available in ORCA
            print('%cpcm\n smd true\n SMDsolvent \"' + solvent + '\"\n end', file=inp_file)
        if optts:
            print(Config.opt_ts_block, file=inp_file)

        if add_bond_ids:
            try:
                [print('%geom\nmodify_internal\n{ B', bond_ids[0], bond_ids[1],  'A } end\nend', file=inp_file)
                 for bond_ids in add_bond_ids]
            except IndexError or TypeError:
                logger.error('Could not add scanned bond')

        if scan_ids:
            try:
                print('%geom Scan\n    B', scan_ids[0], scan_ids[1], '= ' + str(np.round(curr_dist1, 3)) + ', ' +
                      str(np.round(final_dist1, 3)) + ', ' + str(n_steps) + '\n    end\nend', file=inp_file)

                if scan_ids2 is not None:
                    print('%geom Scan\n    B', scan_ids2[0], scan_ids2[1], '= ' + str(np.round(curr_dist2, 3)) + ', ' +
                          str(np.round(final_dist2, 3)) + ', ' + str(n_steps) + '\n    end\nend', file=inp_file)

            except IndexError:
                logger.error('Could not add scan block')

        if distance_constraints:
            print('%geom Constraints', file=inp_file)
            for bond_ids in distance_constraints.keys():
                print('{ B', bond_ids[0], bond_ids[1], distance_constraints[bond_ids], 'C }', file=inp_file)
            print('    end\nend', file=inp_file)

        if len(xyzs) < 33:
            print('%geom MaxIter 100 end', file=inp_file)

        print('%scf \nmaxiter 250 \nend', file=inp_file)
        print('% maxcore', Config.orca_max_core, file=inp_file)
        print('*xyz', charge, mult, file=inp_file)
        [print('{:<3}{:^12.8f}{:^12.8f}{:^12.8f}'.format(*line), file=inp_file) for line in xyzs]
        print('*', file=inp_file)

    return 0


def run_orca(inp_filename, out_filename):
    """
    Run the ORCA calculation given the .inp file as a subprocess
    :param inp_filename:
    :param out_filename:
    :return:
    """
    logger.info('Running ORCA calculation {}'.format(inp_filename))
    if not inp_filename.endswith('.inp'):
        logger.error('ORCA input file needs to end with with .inp')
        return None

    orca_terminated_normally = False

    if os.path.exists(out_filename):
        orca_terminated_normally = did_orca_calculation_terminate_normally(out_filename)

    if not orca_terminated_normally:
        with open(out_filename, 'w') as orca_out:
            orca_run = Popen([Config.path_to_orca, inp_filename], stdout=orca_out)
        orca_run.wait()
        logger.info('ORCA calculation {} done'.format(inp_filename))

    return [line for line in open(out_filename, 'r', encoding="utf-8")]


def did_orca_calculation_terminate_normally(out_filename):

    out_lines = [line for line in open(out_filename, 'r', encoding="utf-8")]
    for n_line, line in enumerate(reversed(out_lines)):
        if 'ORCA TERMINATED NORMALLY' or 'The optimization did not converge' in line:
            logger.info('Found ORCA .out file finished. Will skip the calculation')
            return True
        if n_line > 50:
            # The above lines are pretty close to the end of the file – there's no point parsing it all
            break

    return False


def get_orca_opt_xyzs_energy(out_lines):
    """
    For a lost of ORCA output file lines find the optimised xyzs and energy
    :param out_lines:
    :return:
    """

    opt_converged, geom_section = False, False
    opt_xyzs, energy = [], 0.0

    for line in out_lines:

        if 'THE OPTIMIZATION HAS CONVERGED' in line:
            opt_converged = True
        if 'CARTESIAN COORDINATES' in line and opt_converged:
            geom_section = True

        if geom_section and len(line.split()) == 0:
            geom_section = False

        if geom_section and len(line.split()) == 4:
            atom_label, x, y, z = line.split()
            opt_xyzs.append([atom_label, float(x), float(y), float(z)])

        if 'FINAL SINGLE POINT ENERGY' in line:
            energy = float(line.split()[4])             # e.g. line = 'FINAL SINGLE POINT ENERGY     -4143.815610365798'

    return opt_xyzs, energy


def get_orca_energy(out_lines):
    """
    For a list of ORCA output file lines find the optimised energy
    :param out_lines:
    :return:
    """

    for line in out_lines[::-1]:

        if 'FINAL SINGLE POINT ENERGY' in line:
            return float(line.split()[4])

    logger.error('Could not find orca energy')
    return None


def get_imag_frequencies_xyzs_energy_optts(out_lines):
    """
    For a list of ORCA output lines from an OptTS calculation get the imaginary frequencies (cm-1), final xyz list and
    potential energy
    :param out_lines:
    :return:
    """
    logger.info('Finding imaginary frequencies, xyzs and energy from an ORCA OptTS calculation')

    n_atoms = 0
    freqs, opt_xyzs, energy = [], [], 0.0

    for i, line in enumerate(out_lines):

        if 'Number of atoms' in line:
            n_atoms = int(line.split()[-1])

        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            opt_xyzs = []
            geom_lines = out_lines[i+2:i+1+n_atoms+1]
            for l in geom_lines:
                atom_label, x, y, z = l.split()
                opt_xyzs.append([atom_label, float(x), float(y), float(z)])

        if 'VIBRATIONAL FREQUENCIES' in line:
            freq_lines = out_lines[i+5:i+3*n_atoms+5]
            freqs = [float(l.split()[1]) for l in freq_lines]

        if 'FINAL SINGLE POINT ENERGY' in line:
            energy = float(line.split()[4])

    imag_freqs = [freq for freq in freqs if freq < 0]

    if len(imag_freqs) == 0:
        logger.warning('Found no imaginary frequencies')

    return imag_freqs, opt_xyzs, energy


def get_orca_scan_values_xyzs_energies(out_lines, scan_2d=False):
    """
    For a list of ORCA output lines grab the energies and xyz coordinates at each point and return them as a dictionary
    of [value] = (xyzs, energy)

    a value might be a distance in Å, or an angle in degrees

    :param out_lines:
    :param scan_2d: (bool) Flag for an ORCA 2D scan
    :return:
    """
    logger.info('Getting the xyzs and energies from an ORCA relaxed PES scan')

    values_xyzs_energies = {}
    curr_dist1, curr_dist2, curr_dist = 0, 0, 0
    scan_point_xyzs, scan_point_energy, opt_done, xyz_block = [], 0, False, False

    for n_line, line in enumerate(out_lines):
        if 'The optimization did not converge' in line:
            logger.warning('Optimisation did not converge')
            if scan_2d:
                logger.critical('Can\'t yet handle non-converged 2D scan')
                exit()

            return get_orca_scan1d_values_xyzs_energies_no_conv(out_lines)

        if 'RELAXED SURFACE SCAN STEP' in line:
            scan_point_xyzs, opt_done, xyz_block = [], False, False
            if scan_2d:
                curr_dist1 = float(out_lines[n_line+2].split()[-2])
                curr_dist2 = float(out_lines[n_line+3].split()[-2])
            else:
                curr_dist = float(out_lines[n_line+2].split()[-2])

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

    return values_xyzs_energies


def get_orca_opt_final_xyzs(out_lines):
    """
    For a set of ORCA output lines from a geometry optimisation grab the last set of xyzs
    :param out_lines: (list) xyzs
    :return:
    """

    xyz_section, xyzs = False, []

    for line in out_lines:

        if xyz_section and len(line.split()) == 0:
            xyz_section = False

        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            xyz_section, xyzs = True, []

        if xyz_section and len(line.split()) == 4:
            atom_label, x, y, z = line.split()
            xyzs.append([atom_label, float(x), float(y), float(z)])

    return xyzs


def get_orca_normal_mode_displacements(out_lines, mode_number, n_atoms):
    """
    For a set of ORCA output lines get the displacements along a mode for each of the n_atoms in the structure
    will return a list of length n_atoms each with 3 components (x, y, z)

    :param out_lines: (list) ORCA output lines
    :param mode_number: (int) normal mode number. 6 will be the first vibrational mode (indexed from 0 in ORCA)
    :param n_atoms: (int) number of atoms
    :return:
    """

    normal_mode_section, values_sec, displacements, col = False, False, [], None

    for j, line in enumerate(out_lines):
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
                displacements = [float(disp_line.split()[col]) for disp_line in out_lines[j+1:j+3*n_atoms+1]]

    displacements_xyz = [displacements[i:i + 3] for i in range(0, len(displacements), 3)]
    if len(displacements_xyz) != n_atoms:
        logger.error('Something went wrong getting the displacements n != n_atoms')

    return displacements_xyz


def get_orca_scan1d_values_xyzs_energies_no_conv(out_lines, delta_e_threshold_kcal_mol=1.0):

    logger.info('Getting the xyzs and energies from a non-converged ORCA relaxed PES scan')

    values_xyzs_energies, curr_dist, n_atoms = {}, None, 0
    curr_energy, curr_delta_energy, scan_point_xyzs = 0.0, 0.0, []

    for n_line, line in enumerate(out_lines):
        if 'Number of atoms' in line:
            n_atoms = int(line.split()[-1])

        if 'RELAXED SURFACE SCAN STEP' in line:
            curr_dist = float(out_lines[n_line+2].split()[-2])

        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            scan_point_xyzs = []
            for xyz_line in out_lines[n_line+2:n_line+1+n_atoms+1]:
                atom_label, x, y, z = xyz_line.split()
                scan_point_xyzs.append([atom_label, float(x), float(y), float(z)])

        if 'FINAL SINGLE POINT ENERGY' in line:
            curr_delta_energy = np.abs(float(line.split()[4]) - curr_energy)
            curr_energy = float(line.split()[4])

        if 'RELAXED SURFACE SCAN STEP' in line or 'ORCA TERMINATED NORMALLY' in line:
            if curr_dist is not None and curr_energy != 0.0:
                if Constants.ha2kcalmol * curr_delta_energy < delta_e_threshold_kcal_mol:
                    values_xyzs_energies[curr_dist] = scan_point_xyzs, curr_energy
                else:
                    logger.warning('Optimisation wasn\'t close to converging on this step')

    return values_xyzs_energies
