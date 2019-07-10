import os
import numpy as np
from subprocess import Popen
from .config import Config
from .log import logger
from .constants import Constants


def run_xtb(xyz_filename, opt=True, charge=0, scan_ids=None, solvent=None, curr_dist=1.5, final_dist=3.5, n_steps=20,
            bond_constraints=None, out_filename=None, n_cores=None):
    """
    Run an XTB job using a .xyz file as input. if scan_ids are specified then the distance scan_ids[0] – scan_ids[1]
    will be scanned from curr_dist to final_dist in n_steps. Bond constraints can be specified in a dict of tuples
    containing the atom ids : distance as key, value pairs

    :param xyz_filename: (str)
    :param opt: (bool)
    :param charge: (int)
    :param scan_ids: (tuple)
    :param solvent: (str)
    :param curr_dist: (float)
    :param final_dist: (float)
    :param n_steps: (int)
    :param bond_constraints: (dict)
    :param out_filename: (str)
    :param n_cores: (int) number of cores to run the xtb calculation with
    :return:
    """
    logger.info('Running XTB')

    if not xyz_filename.endswith('.xyz'):
        logger.error('xyz file does not end with .xyz')

    if out_filename is None:
        out_filename = xyz_filename.replace('.xyz', '_xtb.out')

    out_lines = get_out_lines_if_finished(out_filename)
    if out_lines is not None and scan_ids is None and opt:
        return out_lines

    with open(out_filename, 'w') as xtb_out:
        params = [Config.path_to_xtb, xyz_filename, '--chrg', str(charge)]

        if opt:
            params.append('--opt')
        if scan_ids:
            input_filename = 'xcontrol_' + out_filename.replace('.out', '')
            write_scan_input_file(input_filename, scan_ids, curr_dist, final_dist, n_steps, bond_constraints)
            params += ['--input', input_filename]
        if solvent:
            # TODO add check for supported solvents
            params += ['--gbsa', solvent]

        if n_cores is None:
            os.environ['OMP_NUM_THREADS'] = str(Config.n_cores) if Config.n_cores <= 4 else str(4)
        else:
            os.environ['OMP_NUM_THREADS'] = str(n_cores)

        xtb_run = Popen(params, stdout=xtb_out, stderr=open(os.devnull, 'w'))
    xtb_run.wait()

    return [line for line in open(out_filename, 'r')]


def get_xtb_xyzs_energy(out_lines):

    opt_converged, geom_section = False, False
    opt_xyzs = []
    energy = 0.0

    for line in out_lines:
        if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
            opt_converged = True

        if '$coord' in line and opt_converged:
            geom_section = True

        if '$end' in line and geom_section:
            geom_section = False

        if len(line.split()) == 4 and geom_section:
            x, y, z, atom_label = line.split()
            opt_xyzs.append([atom_label,
                             float(x) * Constants.a02ang, float(y) * Constants.a02ang, float(z) * Constants.a02ang])

        if 'total E' in line:
            energy = float(line.split()[-1])
        if 'TOTAL ENERGY' in line:
            energy = float(line.split()[-3])

    return opt_xyzs, energy


def get_out_lines_if_finished(xtb_out_filename):
    """
    If XTB has already run then return the output file lines, else return None
    :param xtb_out_filename:
    :return:
    """

    if os.path.exists(xtb_out_filename):
        out_lines = [line for line in open(xtb_out_filename, 'r')]
        for line in out_lines[::-1]:
            if ' * finished run' in line:
                logger.info('Found XTB calculation done')
                return out_lines

    return None


def write_scan_input_file(input_filename, scan_ids, curr_dist, final_dist, n_steps, bond_constraints):

    with open(input_filename, 'w') as in_file:
        print(('$scan\nmode=sequential\n' +                 # XTB counts from 1 so increment atom ids by 1
              'distance: ' + str(scan_ids[0] + 1) + ', ' + str(scan_ids[1] + 1) + ', ' + str(np.round(curr_dist, 3)) +
               '; ' + str(np.round(curr_dist, 3)) + ',' + str(np.round(final_dist, 3)) + ',' + str(n_steps) + '\n$'),
              file=in_file)

        if bond_constraints:
            for atom_ids in bond_constraints.keys():           # XTB counts from 1 so increment atom ids by 1
                print('$constrain\nforce constant=20\ndistance:' + str(atom_ids[0] + 1) + ', ' + str(atom_ids[1] + 1) +
                      ', ' + str(np.round(bond_constraints[atom_ids], 3)) + '\n$', file=in_file)

    return input_filename


def get_xtb_scan_xyzs_energies(values, filename='xtbscan.log'):

    if not os.path.exists(filename):
        return None

    scan_file_lines = open(filename, 'r').readlines()
    energies, xyzs, values_xyzs_energies = [], [], {}
    try:
        n_atoms = int(scan_file_lines[0].split()[0])
        for j, line in enumerate(scan_file_lines):

            if 'SCF done' in line:
                energies.append(float(line.split()[2]))
                curr_xyzs = []
                for j in range(j+1, j+n_atoms+1):
                    atom_label, x, y, z = scan_file_lines[j].split()
                    curr_xyzs.append([atom_label, float(x), float(y), float(z)])
                xyzs.append(curr_xyzs)

            if j == len(scan_file_lines) - n_atoms:
                if len(values) != len(xyzs) or len(values) != len(energies):
                    logger.error('Didn\'t find the correct number of energies or xyzs from XTB scan')
                else:
                    for j in range(len(values)):
                        values_xyzs_energies[values[j]] = (xyzs[j], energies[j])

                return values_xyzs_energies

    except IndexError:
        return None
