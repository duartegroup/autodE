from copy import deepcopy
import numpy as np
from .config import Config
from .log import logger
from .transition_state import TS
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca
from .ORCAio import get_imag_frequencies_xyzs_energy_optts
from .ORCAio import get_orca_normal_mode_displacements
from .ORCAio import get_orca_opt_final_xyzs


def get_ts(ts_guess_xyzs, charge, mult, solvent, name='reaction_TS', bond_ids_to_add=None, imag_freq_threshold=-50):
    """
    Get a transition object from a set of xyzs by running an ORCA OptTS calculation
    :param ts_guess_xyzs: (list(list)) xyzs
    :param charge: (int) charge on the TS
    :param mult: (int) multiplicity on the TS
    :param solvent: (str) name of the implicit solvent to run the optimisation in
    :param name: (str) name of the TS calculation
    :param bond_ids_to_add: (list(tuple)) list of bond ids to add to the redundant internal coordinates
    :param imag_freq_threshold: (float) Imaginary frequency threshold for a *true* TS, given as as a negative
    i.e. non-complex value
    :return:
    """

    orca_out_lines = get_orca_optts_out(ts_guess_xyzs, charge, mult, solvent, name, bond_ids_to_add)
    imag_freqs, ts_xyzs, ts_energy = get_imag_frequencies_xyzs_energy_optts(orca_out_lines)

    if len(imag_freqs) > 1:
        logger.warning('OptTS calculation returned {} imaginary frequencies'.format(len(imag_freqs)))
        orca_out_lines = do_displacements(orca_out_lines, charge, mult, solvent, name, bond_ids_to_add)

    check_calculation_convergence(orca_out_lines, charge, mult, solvent, name, bond_ids_to_add)
    ts_converged = optts_converged(orca_out_lines)
    ts_nearly_converged = is_optts_nearly_converged(orca_out_lines)

    if ts_converged or ts_nearly_converged:
        imag_freqs, ts_xyzs, ts_energy = get_imag_frequencies_xyzs_energy_optts(orca_out_lines)

        if imag_freqs[0] > imag_freq_threshold:
            logger.warning('Probably haven\'t found the correct TS {} > {} cm-1'.format(imag_freqs[0],
                                                                                        imag_freq_threshold))
        if len(imag_freqs) == 1:
            logger.info('Found TS with 1 imaginary frequency')

        if ts_has_correct_imaginary_vector(orca_out_lines, n_atoms=len(ts_guess_xyzs),
                                           active_atom_pairs=bond_ids_to_add):
            if ts_converged:
                return TS(imag_freqs, ts_xyzs, ts_energy)
            if ts_nearly_converged:
                return TS(imag_freqs, ts_xyzs, ts_energy, converged=False)

    return None


def do_displacements(curr_orca_out_lines, charge, mult, solvent, name, bond_ids_to_add):
    displaced_xyzs = get_displaced_xyzs_along_imaginary_mode(curr_orca_out_lines, displacement_magnitude=1.0)
    orca_out_lines = get_orca_optts_out(displaced_xyzs, charge, mult, solvent, name + 'dis', bond_ids_to_add)
    imag_freqs, ts_xyzs, ts_energy = get_imag_frequencies_xyzs_energy_optts(orca_out_lines)
    check_calculation_convergence(orca_out_lines, charge, mult, solvent, name + 'dis', bond_ids_to_add)

    if len(imag_freqs) > 1:
        logger.warning('OptTS calculation returned {} imaginary frequencies'.format(len(imag_freqs)))
        displaced_xyzs = get_displaced_xyzs_along_imaginary_mode(orca_out_lines, displacement_magnitude=-1.0)
        orca_out_lines = get_orca_optts_out(displaced_xyzs, charge, mult, solvent, name + 'dis2', bond_ids_to_add)
        check_calculation_convergence(orca_out_lines, charge, mult, solvent, name + 'dis2', bond_ids_to_add)

        if len(imag_freqs) > 1:
            logger.error('Couldn\'t remove other imaginary frequencies by displacement')

    return orca_out_lines


def check_calculation_convergence(orca_out_lines, charge, mult, solvent, name, bond_ids_to_add):

    if not optts_converged(orca_out_lines):
        if is_optts_nearly_converged(orca_out_lines):
            logger.info('OptTS nearly did converge. Will try more steps')
            last_xys = get_orca_opt_final_xyzs(orca_out_lines)
            return get_orca_optts_out(last_xys, charge, mult, solvent, name + 'reopt', bond_ids_to_add)

        logger.warning('OptTS calculation was no where near converging')
    return None


def get_orca_optts_out(ts_guess_xyzs, charge, mult, solvent, name, bond_ids_to_add):
    """
    For a reaction given a guess set of xyzs run an ORCA OptTS and return the output lines
    :param name: (str)
    :param charge: (int)
    :param mult: (int)
    :param solvent: (str)
    :param ts_guess_xyzs: (list)
    :param bond_ids_to_add: (list)
    :return:
    """
    logger.info('Getting ORCA out lines from OptTS calculation')

    orca_inp_filename = name + '_orca_opt.inp'
    gen_orca_inp(orca_inp_filename, Config.opt_ts_keywords, ts_guess_xyzs, charge, mult, solvent, Config.n_cores,
                 add_bond_ids=bond_ids_to_add, optts=True)
    orca_out_lines = run_orca(orca_inp_filename, out_filename=orca_inp_filename.replace('.inp', '.out'))

    return orca_out_lines


def get_displaced_xyzs_along_imaginary_mode(out_lines, mode_number=7, displacement_magnitude=1.0):
    """
    Displace the geometry along the imaginary mode with mode number iterating from 0, where 0-2 are translational
    normal modes, 3-5 are rotational modes and 6 is the largest imaginary mode. To displace along the second imaginary
    mode we have mode_number=7
    :param out_lines: (list) ORCA out lines
    :param mode_number: (int)
    :param displacement_magnitude: (float)
    :return: (list)
    """
    logger.info('Displacing along imaginary mode')

    current_xyzs = get_orca_opt_final_xyzs(out_lines)
    n_atoms = len(current_xyzs)
    mode_distplacement_coords = get_orca_normal_mode_displacements(out_lines, mode_number, n_atoms)

    displaced_xyzs = deepcopy(current_xyzs)
    for i in range(n_atoms):
        for j in range(3):
            displaced_xyzs[i][j+1] += displacement_magnitude * mode_distplacement_coords[i][j]      # adding coord (nx3)
    #                                                                                               # to xyzs (nx4)
    return displaced_xyzs


def optts_converged(out_lines):
    for line in out_lines[::-1]:
        if 'THE OPTIMIZATION HAS CONVERGED' in line:
            logger.info('OptTS geometry converged')
            return True
    logger.warning('OptTS geometry *not* converged')
    return False


def is_optts_nearly_converged(out_lines):
    """
    Check whether an ORCA calculation has nearly converged and may just need more geometry optimisation steps to
    complete successfully
    :param out_lines: (list) orca out lines
    :return: (bool)
    """
    geom_conv_block, n_yess = False, 0

    for line in out_lines[::-1]:
        if geom_conv_block and 'Geometry convergence' in line:
            geom_conv_block = False
        if 'The optimization has not yet converged' in line:
            geom_conv_block = True
        if geom_conv_block and len(line.split()) == 5:
            if line.split()[-1] == 'YES':
                return True

    return False


def ts_has_correct_imaginary_vector(out_lines, n_atoms, active_atom_pairs, threshold_contribution=0.3):
    """
    For an orca output file check that the first imaginary mode (number 6) in the final frequency calculation
    contains the correct motion, i.e. contributes more than threshold_contribution in relative terms to the
    magnitude of the sum of the forces

    :param out_lines: (list)
    :param n_atoms: (int) number of atoms
    :param active_atom_pairs: (list(tuples))
    :param threshold_contribution: (float) threshold contribution to the imaginary mode from the atoms in
    bond_ids_to_add
    :return:
    """
    logger.info('Checking the active atoms contribute more than {} to the imag mode'.format(threshold_contribution))

    if active_atom_pairs is None:
        logger.info('Cannot determine whether the correct atoms contribute')
        return True

    imag_normal_mode_displacements_xyz = get_orca_normal_mode_displacements(out_lines, mode_number=6, n_atoms=n_atoms)
    imag_mode_magnitudes = [np.linalg.norm(np.array(dis_xyz)) for dis_xyz in imag_normal_mode_displacements_xyz]

    should_be_active_atom_magnitues = []
    for atom_id in range(n_atoms):
        if any([atom_id in bond_ids for bond_ids in active_atom_pairs]):
            should_be_active_atom_magnitues.append(imag_mode_magnitudes[atom_id])

    relative_contribution = np.sum(np.array(should_be_active_atom_magnitues)) / np.sum(np.array(imag_mode_magnitudes))
    if relative_contribution > threshold_contribution:
        logger.info('TS has significant contribution from the active atoms to the imag mode')
        return True

    logger.info('TS has *no* significant contribution from the active atoms to the imag mode')
    return False
