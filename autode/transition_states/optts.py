from copy import deepcopy
import numpy as np
from autode.log import logger
from autode.transition_states.transition_state import TS


def get_ts(ts_guess, imag_freq_threshold=-100):
    """
    Get a transition object from a set of xyzs by running an ORCA OptTS calculation
    :param ts_guess: (object) TSguess object
    :param imag_freq_threshold: (float) Imaginary frequency threshold for a *true* TS, given as as a negative
    i.e. non-complex value
    :return:
    """

    if ts_guess is None:
        logger.warning('Cannot find a transition state; had no TS guess')
        return None

    ts_guess.run_orca_optts()
    imag_freqs, ts_xyzs, ts_energy = ts_guess.get_imag_frequencies_xyzs_energy()

    if not ts_has_correct_imaginary_vector(ts_guess.optts_calc, n_atoms=len(ts_guess.xyzs),
                                           active_bonds=ts_guess.active_bonds):
        return None

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
                return None

            if len(imag_freqs) == 1:
                logger.info('Found TS with 1 imaginary frequency')

            if ts_has_correct_imaginary_vector(ts_guess.optts_calc, n_atoms=len(ts_guess.xyzs),
                                               active_bonds=ts_guess.active_bonds):

                if ts_guess.optts_converged:
                    return TS(ts_guess)

                if ts_guess.optts_nearly_converged:
                    return TS(ts_guess, converged=False)
        else:
            logger.warning('TS has *0* imaginary frequencies')

    return None


def get_displaced_xyzs_along_imaginary_mode(calc, mode_number=7, displacement_magnitude=1.0):
    """
    Displace the geometry along the imaginary mode with mode number iterating from 0, where 0-2 are translational
    normal modes, 3-5 are rotational modes and 6 is the largest imaginary mode. To displace along the second imaginary
    mode we have mode_number=7
    :param calc: (object)
    :param mode_number: (int)
    :param displacement_magnitude: (float)
    :return: (list)
    """
    logger.info('Displacing along imaginary mode')

    current_xyzs = calc.get_final_xyzs()
    n_atoms = len(current_xyzs)
    mode_distplacement_coords = calc.get_normal_mode_displacements(mode_number=mode_number)

    displaced_xyzs = deepcopy(current_xyzs)
    for i in range(n_atoms):
        for j in range(3):
            displaced_xyzs[i][j+1] += displacement_magnitude * mode_distplacement_coords[i][j]      # adding coord (nx3)
    #                                                                                               # to xyzs (nx4)
    return displaced_xyzs


def ts_has_correct_imaginary_vector(calc, n_atoms, active_bonds, threshold_contribution=0.25):
    """
    For an orca output file check that the first imaginary mode (number 6) in the final frequency calculation
    contains the correct motion, i.e. contributes more than threshold_contribution in relative terms to the
    magnitude of the sum of the forces

    :param calc: (object)
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

    imag_normal_mode_displacements_xyz = calc.get_normal_mode_displacements(mode_number=6)
    if imag_normal_mode_displacements_xyz is None:
        logger.error('Have no imaginary normal mode displacements to analyse')
        return False

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
