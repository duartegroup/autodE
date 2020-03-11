from copy import deepcopy
import numpy as np
from autode.config import Config
from autode.log import logger
from autode.atoms import get_atomic_weight
from autode.molecule import Molecule
from autode.calculation import Calculation
from autode.mol_graphs import is_isomorphic
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm


def get_ts(ts_guess, solvent_mol=None, imag_freq_threshold=-100):
    """Get a transition state object from a set of xyzs by running an ORCA OptTS calculation

    Arguments:
        ts_guess (ts guess object): object to be optimised to a TS

    Keyword Arguments:
        solvent_mol (mol obj): Solvent molecule object, if none qmmm has not been done, will not include point charges in optts
        imag_freq_threshold (float): Imaginary frequency threshold for a *true* TS, given as as a negative i.e. non-complex value (default: {-100})

    Returns:
        tuple: (TS object, bool saying if true TS or not)
    """

    if ts_guess is None:
        logger.warning('Cannot find a transition state; had no TS guess')
        return None, False

    if solvent_mol:
        _, qmmm_xyzs, n_qm_atoms = do_explicit_solvent_qmmm(ts_guess, solvent_mol, get_hmethod())
        ts_guess.qm_solvent_xyzs = qmmm_xyzs[ts_guess.n_atoms:n_qm_atoms]
        ts_guess.mm_solvent_xyzs = qmmm_xyzs[n_qm_atoms:]
        point_charges = []
        for i, xyz in enumerate(ts_guess.mm_solvent_xyzs):
            point_charges.append(xyz + [solvent_mol.charges[i % solvent_mol.n_atoms]])
        ts_guess.point_charges = point_charges

    ts_guess.run_optts()
    if ts_guess.calc_failed:
        return None, False

    ts_guess.check_optts_convergence()
    if ts_guess.calc_failed:
        return None, False

    imag_freqs, _, _ = ts_guess.get_imag_frequencies_xyzs_energy()

    if not ts_has_correct_imaginary_vector(ts_guess.optts_calc, n_atoms=ts_guess.n_atoms, active_bonds=ts_guess.active_bonds, molecules=(ts_guess.reactant, ts_guess.product)):
        return None, False

    if len(imag_freqs) > 1:
        logger.warning(f'OptTS calculation returned {len(imag_freqs)} imaginary frequencies')
        ts_guess.do_displacements()
        if ts_guess.calc_failed:
            return None, False

    ts_guess.check_optts_convergence()
    if ts_guess.calc_failed:
        return None, False

    if ts_guess.optts_converged or ts_guess.optts_nearly_converged:
        imag_freqs, _, _ = ts_guess.get_imag_frequencies_xyzs_energy()
        if len(imag_freqs) > 0:

            if imag_freqs[0] > imag_freq_threshold:
                logger.warning(f'Probably haven\'t found the correct TS {imag_freqs[0]} > {imag_freq_threshold} cm-1')
                return None, False

            if len(imag_freqs) == 1:
                logger.info('Found TS with 1 imaginary frequency')

            if ts_has_correct_imaginary_vector(ts_guess.optts_calc, n_atoms=ts_guess.n_atoms, active_bonds=ts_guess.active_bonds, molecules=(ts_guess.reactant, ts_guess.product)):

                if ts_guess.optts_converged:
                    return ts_guess, True

                if ts_guess.optts_nearly_converged:
                    return ts_guess, False
        else:
            logger.warning('TS has *0* imaginary frequencies')

    return None, False


def get_displaced_xyzs_along_imaginary_mode(calc, n_atoms, mode_number=7, displacement_magnitude=1.0):
    """Displace the geometry along the imaginary mode with mode number iterating from 0, where 0-2 are translational
    normal modes, 3-5 are rotational modes and 6 is the largest imaginary mode. To displace along the second imaginary
    mode we have mode_number=7

    Arguments:
        calc (calc object): calculation object
        n_atoms (int): number of solute atoms

    Keyword Arguments:
        mode_number (int): mode number to displace along (default: {7})
        displacement_magnitude (float): distance to displace (default: {1.0})

    Returns:
        list(list): list of xyzs
    """
    logger.info('Displacing along imaginary mode')

    current_xyzs = calc.get_final_xyzs()[:n_atoms]
    mode_distplacement_coords = calc.get_normal_mode_displacements(mode_number=mode_number)

    displaced_xyzs = deepcopy(current_xyzs)
    for i in range(len(mode_distplacement_coords)):
        for j in range(3):
            displaced_xyzs[i][j+1] += displacement_magnitude * mode_distplacement_coords[i][j]      # adding coord (nx3)
            # to xyzs (nx4)
    return displaced_xyzs


def ts_has_correct_imaginary_vector(calc, n_atoms, active_bonds, molecules=None, threshold_contribution=0.25):
    """For an orca output file check that the first imaginary mode (number 6) in the final frequency calculation
    contains the correct motion, i.e. contributes more than threshold_contribution in relative terms to the
    magnitude of the sum of the forces

    Arguments:
        calc (calc object): calculation object
        n_atoms (int): number of solute atoms
        active_bonds (list(tuples)): bonds being made/broken

    Keyword Arguments:
        molecules (list): list of reactant and product molecule objects (default: {None})
        threshold_contribution (float): threshold contribution to the imaginary mode from the atoms in

    Returns:
        bool: if the imaginary mode is correct or not
    """

    logger.info(f'Checking the active atoms contribute more than {threshold_contribution} to the imag mode')

    if active_bonds is None:
        logger.info('Cannot determine whether the correct atoms contribute')
        return True

    imag_normal_mode_displacements_xyz = calc.get_normal_mode_displacements(mode_number=6)
    if imag_normal_mode_displacements_xyz is None:
        logger.error('Have no imaginary normal mode displacements to analyse')
        return False

    final_xyzs = calc.get_final_xyzs()[:n_atoms]

    imag_mode_magnitudes = [np.linalg.norm(np.array(dis_xyz)) for dis_xyz in imag_normal_mode_displacements_xyz]

    weighted_imag_mode_magnitudes = []

    for index, atom in enumerate(final_xyzs):
        atom_label = atom[0]
        weighting = get_atomic_weight(atom_label) + 10
        weighted_imag_mode_magnitudes.append((imag_mode_magnitudes[index] * weighting))

    should_be_active_atom_magnitudes = []
    for atom_id in range(n_atoms):
        if any([atom_id in bond_ids for bond_ids in active_bonds]):
            should_be_active_atom_magnitudes.append(weighted_imag_mode_magnitudes[atom_id])

    relative_contribution = np.sum(np.array(should_be_active_atom_magnitudes)) / np.sum(np.array(weighted_imag_mode_magnitudes))

    if molecules is not None:
        if threshold_contribution - 0.1 < relative_contribution < threshold_contribution + 0.1:
            logger.info(f'Unsure if significant contribution from active atoms to imag mode '
                        f'(contribution = {relative_contribution:.3f}). Displacing along imag modes to check')

            if check_close_imag_contribution(calc, n_atoms, molecules, method=get_lmethod()):
                logger.info('Imaginary mode links reactants and products, TS found')
                return True
            logger.info('Lower level method didn\'t find link, trying higher level of theory')
            if check_close_imag_contribution(calc, n_atoms, molecules, method=get_hmethod()):
                logger.info('Imaginary mode links reactants and products, TS found')
                return True
            logger.info('Imaginary mode doesn\'t link reactants and products, TS *not* found')
            return False

    if relative_contribution > threshold_contribution:
        logger.info(f'TS has significant contribution from the active atoms to the imag mode '
                    f'(contribution = {relative_contribution:.3f})')
        return True

    logger.info(f'TS has *no* significant contribution from the active atoms to the imag mode '
                f'(contribution = {relative_contribution:.3f})')
    return False


def check_close_imag_contribution(calc, n_atoms, molecules, method, disp_mag=1):
    """Displaces atoms along the imaginary mode to see if products and reactants are made

    Arguments:
        calc (calculation obj): calculation oobject
        n_atoms (int): number of solute atoms
        molecules (tuple): tuple containing the reactant and product objects
        method (electronic structure method): electronic structure method to use


    Keyword Arguments:
        disp_mag (int): Distance to be displaced along the imag mode (default: {1})

    Returns:
        bool: if the imag mode is correct or not
    """
    forward_displaced_xyzs = get_displaced_xyzs_along_imaginary_mode(calc, n_atoms, mode_number=6, displacement_magnitude=disp_mag)
    forward_displaced_mol = Molecule(xyzs=forward_displaced_xyzs[:n_atoms], charge=calc.charge, mult=calc.mult)
    forward_coords = forward_displaced_mol.get_coords()
    forward_distance_constraints = {}

    for active_bond in calc.bond_ids_to_add:
        distance = np.linalg.norm(forward_coords[active_bond[0]] - forward_coords[active_bond[1]])
        forward_distance_constraints[active_bond] = distance
    forward_displaced_calc = Calculation(name=calc.name + '_forwards_displacement', molecule=forward_displaced_mol, method=method,
                                         keywords=method.opt_keywords, n_cores=Config.n_cores,
                                         max_core_mb=Config.max_core, opt=True, distance_constraints=forward_distance_constraints, constraints_already_met=True)
    forward_displaced_calc.run()
    forward_displaced_mol.set_xyzs(forward_displaced_calc.get_final_xyzs())

    backward_displaced_xyzs = get_displaced_xyzs_along_imaginary_mode(calc, n_atoms, mode_number=6, displacement_magnitude=-disp_mag)
    backward_displaced_mol = Molecule(xyzs=backward_displaced_xyzs[:n_atoms], charge=calc.charge, mult=calc.mult)
    backward_coords = backward_displaced_mol.get_coords()
    backward_distance_constraints = {}

    for active_bond in calc.bond_ids_to_add:
        distance = np.linalg.norm(backward_coords[active_bond[0]] - backward_coords[active_bond[1]])
        backward_distance_constraints[active_bond] = distance

    backward_displaced_calc = Calculation(name=calc.name + '_backwards_displacement', molecule=backward_displaced_mol, method=method,
                                          keywords=method.opt_keywords, n_cores=Config.n_cores,
                                          max_core_mb=Config.max_core, opt=True, distance_constraints=backward_distance_constraints, constraints_already_met=True)
    backward_displaced_calc.run()
    backward_displaced_mol.set_xyzs(backward_displaced_calc.get_final_xyzs())

    if is_isomorphic(forward_displaced_mol.graph, molecules[0].graph):
        if is_isomorphic(backward_displaced_mol.graph, molecules[1].graph):
            return True
    if is_isomorphic(backward_displaced_mol.graph, molecules[0].graph):
        if is_isomorphic(forward_displaced_mol.graph, molecules[1].graph):
            return True
    return False
