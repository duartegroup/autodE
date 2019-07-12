from copy import deepcopy
import numpy as np
from .log import logger
from .transition_state import TS
from .ORCAio import get_orca_normal_mode_displacements
from .ORCAio import get_orca_opt_final_xyzs


def get_ts(ts_guess, imag_freq_threshold=-50):
    """
    Get a transition object from a set of xyzs by running an ORCA OptTS calculation
    :param ts_guess: (object) TSguess object
    :param imag_freq_threshold: (float) Imaginary frequency threshold for a *true* TS, given as as a negative
    i.e. non-complex value
    :return:
    """

    ts_guess.run_orca_optts()
    imag_freqs, ts_xyzs, ts_energy = ts_guess.get_imag_frequencies_xyzs_energy()

    if len(imag_freqs) > 1:
        logger.warning('OptTS calculation returned {} imaginary frequencies'.format(len(imag_freqs)))
        ts_guess.do_displacements()

    ts_guess.check_optts_convergence()

    if ts_guess.optts_converged or ts_guess.optts_nearly_converged:
        imag_freqs, ts_xyzs, ts_energy = ts_guess.get_imag_frequencies_xyzs_energy()
        if len(imag_freqs) > 0:

            if imag_freqs[0] > imag_freq_threshold:
                logger.warning('Probably haven\'t found the correct TS {} > {} cm-1'.format(imag_freqs[0],
                                                                                            imag_freq_threshold))
            if len(imag_freqs) == 1:
                logger.info('Found TS with 1 imaginary frequency')

            if ts_has_correct_imaginary_vector(ts_guess.optts_out_lines, n_atoms=len(ts_guess.xyzs),
                                               active_bonds=ts_guess.active_bonds):

                if ts_guess.optts_converged:
                    return TS(ts_guess)

                if ts_guess.optts_nearly_converged:
                    return TS(ts_guess, converged=False)

    return None


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


def ts_has_correct_imaginary_vector(out_lines, n_atoms, active_bonds, threshold_contribution=0.25):
    """
    For an orca output file check that the first imaginary mode (number 6) in the final frequency calculation
    contains the correct motion, i.e. contributes more than threshold_contribution in relative terms to the
    magnitude of the sum of the forces

    :param out_lines: (list)
    :param n_atoms: (int) number of atoms
    :param active_bonds: (list(tuples))
    :param threshold_contribution: (float) threshold contribution to the imaginary mode from the atoms in
    bond_ids_to_add
    :return:
    """
    logger.info('Checking the active atoms contribute more than {} to the imag mode'.format(threshold_contribution))

    if active_bonds is None:
        logger.info('Cannot determine whether the correct atoms contribute')
        return True

    imag_normal_mode_displacements_xyz = get_orca_normal_mode_displacements(out_lines, mode_number=6, n_atoms=n_atoms)
    imag_mode_magnitudes = [np.linalg.norm(np.array(dis_xyz)) for dis_xyz in imag_normal_mode_displacements_xyz]

    should_be_active_atom_magnitudes = []
    for atom_id in range(n_atoms):
        if any([atom_id in bond_ids for bond_ids in active_bonds]):
            should_be_active_atom_magnitudes.append(imag_mode_magnitudes[atom_id])

    relative_contribution = np.sum(np.array(should_be_active_atom_magnitudes)) / np.sum(np.array(imag_mode_magnitudes))
    if relative_contribution > threshold_contribution:
        logger.info('TS has significant contribution from the active atoms to the imag mode')
        return True

    logger.info('TS has *no* significant contribution from the active atoms to the imag mode')
    return False
