import os
import numpy as np
from .constants import Constants
from subprocess import Popen
from .log import logger
from .wrappers.wrappers import ORCA
from .wrappers.wrappers import XTB
from .wrappers.wrappers import MOPAC

# TODO implement MOPAC here


class Calculation:

    def get_energy(self):
        logger.info('Getting energy from {}'.format(self.output_filename))
        if self.terminated_normally:
            for line in list(reversed(self.output_file_lines)):

                if self.method == ORCA:
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        return float(line.split()[4])

                if self.method == XTB:
                    if 'total E' in line:
                        return float(line.split()[-1])
                    if 'TOTAL ENERGY' in line:
                        return float(line.split()[-3])

                if self.method == MOPAC:
                    raise NotImplementedError

        return None

    def optimisation_converged(self):
        logger.info('Checking to see if the geometry converged')

        for line in list(reversed(self.output_file_lines)):
            if self.method == ORCA:
                if 'THE OPTIMIZATION HAS CONVERGED' in line:
                    return True

            if self.method == XTB:
                if 'GEOMETRY OPTIMIZATION CONVERGED' in line:
                    return True

            if self.method == MOPAC:
                raise NotImplementedError

        return False

    def get_final_xyzs(self):
        logger.info('Getting final xyzs from {}'.format(self.output_filename))

        xyzs = []

        if self.method == XTB:
            geom_section = False
            for line in self.output_file_lines:
                if '$coord' in line:
                    geom_section = True

                if '$end' in line and geom_section:
                    geom_section = False

                if len(line.split()) == 4 and geom_section:
                    x, y, z, atom_label = line.split()
                    xyzs.append([atom_label, float(x) * Constants.a02ang,
                                 float(y) * Constants.a02ang, float(z) * Constants.a02ang])

        if self.method == ORCA:
            xyz_section = False

            for line in list(reversed(self.output_file_lines)):

                if 'CARTESIAN COORDINATES (A.U.)' in line:
                    xyz_section = True

                if 'CARTESIAN COORDINATES (ANGSTROEM)' in line and xyz_section:
                    xyz_section = False

                if xyz_section and len(line.split()) == 4:
                    atom_label, x, y, z = line.split()
                    xyzs.append([atom_label, float(x), float(y), float(z)])

        if self.method == MOPAC:
            raise NotImplementedError

        return xyzs

    def get_scan_values_xyzs_energies(self):

        if self.method == ORCA:
            logger.info('Getting the xyzs and energies from an ORCA relaxed PES scan')
            scan_2d = True if self.scan_ids2 is not None else False

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

            for n_line, line in enumerate(self.output_file_lines):
                if 'The optimization did not converge' in line:
                    logger.warning('Optimisation did not converge')
                    return get_orca_scan_values_xyzs_energies_no_conv(self.output_file_lines, scan_2d=scan_2d)

                if 'RELAXED SURFACE SCAN STEP' in line:
                    scan_point_xyzs, opt_done, xyz_block = [], False, False
                    if scan_2d:
                        curr_dist1 = float(self.output_file_lines[n_line + 2].split()[-2])
                        curr_dist2 = float(self.output_file_lines[n_line + 3].split()[-2])
                    else:
                        curr_dist = float(self.output_file_lines[n_line + 2].split()[-2])

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

        else:
            raise NotImplementedError

    def calculation_terminated_normally(self):
        logger.info('Checking to see if {} terminated normally'.format(self.output_filename))

        out_lines = [line for line in open(self.output_filename, 'r', encoding="utf-8")]

        if self.method == ORCA:
            for n_line, line in enumerate(reversed(out_lines)):
                if 'ORCA TERMINATED NORMALLY' or 'The optimization did not converge' in line:
                    logger.info('Found ORCA .out file finished. Will skip the calculation')
                    return True
                if n_line > 50:
                    # The above lines are pretty close to the end of the file – there's no point parsing it all
                    return False

        if self.method == XTB:
            for n_line, line in enumerate(reversed(out_lines)):
                if 'ERROR' in line:
                    return False
                if n_line > 10:
                    # With XTB we will search for there being no '#ERROR!' in the last few lines
                    return True

        if self.method == MOPAC:
            raise NotImplementedError

        return False

    def generate_input(self):
        logger.info('Generating input file for {}'.format(self.name))

        if self.method == ORCA:

            self.input_filename = self.name + '_orca.inp'
            self.output_filename = self.name + '_orca.out'

            if len(self.xyzs) == 1:
                for keyword in self.keywords:
                    if keyword.lower() == 'opt' or keyword.lower() == 'looseopt' or keyword.lower() == 'tightopt':
                        logger.warning('Cannot do an optimisation for a single atom')
                        self.keywords.remove(keyword)

            with open(self.input_filename, 'w') as inp_file:
                print('!', *self.keywords, file=inp_file)

                if self.solvent:
                    print('%cpcm\n smd true\n SMDsolvent \"' + self.solvent + '\"\n end', file=inp_file)

                if self.optts_block:
                    print(self.optts_block, file=inp_file)

                if self.bond_ids_to_add:
                    try:
                        [print('%geom\nmodify_internal\n{ B', bond_ids[0], bond_ids[1], 'A } end\nend', file=inp_file)
                         for bond_ids in self.bond_ids_to_add]
                    except IndexError or TypeError:
                        logger.error('Could not add scanned bond')

                if self.scan_ids:
                    try:
                        print('%geom Scan\n    B', self.scan_ids[0], self.scan_ids[1],
                              '= ' + str(np.round(self.curr_d1, 3)) + ', ' +
                              str(np.round(self.final_d1, 3)) + ', ' + str(self.n_steps) + '\n    end\nend',
                              file=inp_file)

                        if self.scan_ids2 is not None:
                            print('%geom Scan\n    B', self.scan_ids2[0], self.scan_ids2[1],
                                  '= ' + str(np.round(self.curr_d2, 3)) + ', ' +
                                  str(np.round(self.final_d2, 3)) + ', ' + str(self.n_steps) + '\n    end\nend',
                                  file=inp_file)

                    except IndexError:
                        logger.error('Could not add scan block')

                if self.distance_constraints:
                    print('%geom Constraints', file=inp_file)
                    for bond_ids in self.distance_constraints.keys():
                        print('{ B', bond_ids[0], bond_ids[1], self.distance_constraints[bond_ids], 'C }',
                              file=inp_file)
                    print('    end\nend', file=inp_file)

                if len(self.xyzs) < 33:
                    print('%geom MaxIter 100 end', file=inp_file)

                if self.n_cores > 1:
                    print('%pal nprocs' + str(self.n_cores) + '\nend', file=inp_file)
                print('%scf \nmaxiter 250 \nend', file=inp_file)
                print('% maxcore', self.max_core_mb, file=inp_file)
                print('*xyz', self.charge, self.mult, file=inp_file)
                [print('{:<3}{:^12.8f}{:^12.8f}{:^12.8f}'.format(*line), file=inp_file) for line in self.xyzs]
                print('*', file=inp_file)

        if self.method == XTB:
            self.input_filename = self.name + '_xtb.xyz'
            self.output_filename = self.name + '_xtb.out'

            # Add
            self.flags = ['--chrg', str(self.charge)]

            if self.opt:
                self.flags.append('--opt')

            if self.solvent:
                self.flags += ['--gbsa', self.solvent]

            if self.distance_constraints:
                xcontrol_filename = 'xcontrol_' + self.name
                with open(xcontrol_filename, 'w') as xcontrol_file:
                    for atom_ids in self.distance_constraints.keys():     # XTB counts from 1 so increment atom ids by 1
                        print('$constrain\nforce constant=20\ndistance:' + str(atom_ids[0] + 1) + ', ' + str(
                            atom_ids[1] + 1) + ', ' + str(np.round(self.distance_constraints[atom_ids], 3)) + '\n$',
                              file=xcontrol_file)

                self.flags += ['--input', xcontrol_filename]

            if self.scan_ids or self.scan_ids2:
                logger.critical('Cannot run an XTB 1D or 2D scan. Use constrained optimisations instead')
                exit()

        if self.method == MOPAC:
            raise NotImplementedError

        return None

    def execute_calculation(self):
        logger.info('Running calculation {}'.format(self.input_filename))

        if self.input_filename is None:
            logger.error('Could not run the calculation. Input filename not defined')
            return

        if not os.path.exists(self.input_filename):
            logger.error('Could not run the calculation. Input file does not exist')
            return

        if os.path.exists(self.output_filename):
            self.output_file_exists = True

        if self.output_file_exists:
            if self.calculation_terminated_normally():
                logger.info('Calculated already terminated successfully. Skipping')
                self.output_file_lines = [line for line in open(self.output_filename, 'r', encoding="utf-8")]
                return

        if self.method == XTB:
            logger.info('Setting the number of OMP threads to {}'.format(self.n_cores))
            os.environ['OMP_NUM_THREADS'] = str(self.n_cores)

        with open(self.output_filename, 'w') as output_file:

            params = [self.method.path, self.input_filename]
            if self.flags is not None:
                params += self.flags

            subprocess = Popen(params, stdout=output_file, stderr=open(os.devnull, 'w'))
        subprocess.wait()

        self.output_file_lines = [line for line in open(self.output_filename, 'r', encoding="utf-8")]
        logger.info('Calculation {} done'.format(self.output_filename))

        return None

    def run(self):
        logger.info('Running calculation of {}'.format(self.name))

        self.generate_input()
        self.execute_calculation()
        self.terminated_normally = self.calculation_terminated_normally()

        return None

    def __init__(self, name, xyzs, method, keywords, charge=0, mult=1, solvent=None, n_cores=1, max_core_mb=1000,
                 bond_ids_to_add=None, optts_block=False, scan_ids=None, curr_dist1=1.5, final_dist1=3.5, opt=False,
                 curr_dist2=1.5, final_dist2=3.5, n_steps=10, scan_ids2=None, distance_constraints=None):
        """
        :param xyzs: (list(list))
        :param charge: (int)
        :param mult: (int)
        :param solvent: (str)
        :param n_cores: (int)
        :param bond_ids_to_add: (list(tuples))
        :param optts_block: (bool)
        :param scan_ids: (tuple)
        :param curr_dist1: (float)
        :param final_dist1: (float)
        :param curr_dist2: (float)
        :param final_dist2: (float)
        :param n_steps: (int)
        :param scan_ids2: (tuple)
        :param distance_constraints: (dict) keys: tuple of atom ids (indexed from 0), values: float of the distance
        """

        self.name = name
        self.xyzs = xyzs

        if xyzs is None or len(xyzs) == 0:
            logger.error('Have no xyzs. Can\'run a calculation')
            return

        self.charge = charge
        self.mult = mult
        self.method = method
        self.keywords = keywords
        self.flags = None
        self.opt = opt

        if solvent.lower() not in method.aval_solvents:                                 # Lowercase everything
            logger.critical('Solvent is not available. Cannot run the calculation')
            exit()

        self.solvent = solvent

        self.n_cores = n_cores
        self.max_core_mb = max_core_mb                                                  # Maximum memory per core to use

        self.bond_ids_to_add = bond_ids_to_add
        self.optts_block = optts_block
        self.scan_ids = scan_ids
        self.curr_d1 = curr_dist1
        self.final_d1 = final_dist1
        self.curr_d2 = curr_dist2
        self.final_d2 = final_dist2
        self.n_steps = n_steps
        self.scan_ids2 = scan_ids2
        self.distance_constraints = distance_constraints

        self.input_filename = None
        self.output_filename = None

        self.output_file_exists = False
        self.terminated_normally = False
        self.output_file_lines = None

