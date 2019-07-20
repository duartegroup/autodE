import numpy as np
from .config import Config
from .log import logger
from .XTBio import run_xtb
from .XTBio import get_xtb_scan_xyzs_energies
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca
from .ORCAio import get_orca_scan_values_xyzs_energies
from .input_output import xyzs2xyzfile
from .ts_guess import TSguess


def get_orca_ts_guess_1dpes_scan(mol, active_bond, n_steps, orca_keywords, name, reaction_class, delta_dist=1.5,
                                 active_bonds_not_scanned=None):
    """
    Scan the distance between 2 atoms and return the xyzs with peak energy
    :param mol: Molecule object
    :param active_bond: (tuple) of atom ids
    :param delta_dist: (float) Distance to add onto the current distance (Å)
    :param n_steps: (int) Number of scan steps to use in the XTB scan
    :param orca_keywords: (list) ORCA keywords to use
    :param name: (str)
    :param reaction_class: (object) class of the reaction (reactions.py)
    :param active_bonds_not_scanned: list(tuple) pairs of atoms that are active, but will not be scanned in the 1D PES
    :return: TSguess object
    """
    logger.info('Getting TS guess from ORCA relaxed potential energy scan')

    curr_dist = mol.distance_matrix[active_bond[0], active_bond[1]]
    final_dist = curr_dist + delta_dist

    scan_inp_filename = name + '_orca_scan.inp'
    gen_orca_inp(scan_inp_filename, orca_keywords, mol.xyzs, mol.charge, mol.mult, mol.solvent, Config.n_cores,
                 scan_ids=active_bond, curr_dist1=curr_dist, final_dist1=final_dist, n_steps=n_steps)

    orca_out_lines = run_orca(scan_inp_filename, out_filename=scan_inp_filename.replace('.inp', '.out'))
    dist_xyzs_energies = get_orca_scan_values_xyzs_energies(orca_out_lines)
    ts_guess_xyzs = find_1dpes_maximum_energy_xyzs(dist_xyzs_energies)

    active_bonds = [active_bond] if active_bonds_not_scanned is None else [active_bond] + active_bonds_not_scanned

    return TSguess(name=name, reaction_class=reaction_class, xyzs=ts_guess_xyzs, solvent=mol.solvent,
                   charge=mol.charge, mult=mol.mult, active_bonds=active_bonds)


def get_xtb_ts_guess_1dpes_scan(mol, active_bond, n_steps, reaction_class, delta_dist=1.5, active_bonds_not_scanned=None):
    """
    Scan the distance between 2 atoms and return the xyzs with peak energy
    :param mol: Molecule object
    :param active_bond: (tuple) of atom ids
    :param delta_dist: (float) Distance to add onto the current distance (Å)
    :param n_steps: (int) Number of scan steps to use in the XTB scan
    :param reaction_class: (object) class of the reaction (reactions.py)
    :param active_bonds_not_scanned: list(tuple) pairs of atoms that are active, but will not be scanned in the 1D PES
    :return: List of xyzs
    """
    logger.info('Getting TS guess from XTB relaxed potential energy scan')

    curr_dist = mol.distance_matrix[active_bond[0], active_bond[1]]
    final_dist = curr_dist + delta_dist

    reac_xyz_filename = xyzs2xyzfile(mol.xyzs, basename=mol.name)
    run_xtb(reac_xyz_filename, charge=mol.charge, scan_ids=active_bond, solvent=mol.solvent, curr_dist=curr_dist,
            final_dist=final_dist, n_steps=n_steps)
    dist_xyzs_energies = get_xtb_scan_xyzs_energies(values=np.linspace(curr_dist, final_dist, n_steps))
    ts_guess_xyzs = find_1dpes_maximum_energy_xyzs(dist_xyzs_energies)

    active_bonds = [active_bond] if active_bonds_not_scanned is None else [active_bond] + active_bonds_not_scanned

    return TSguess(reaction_class=reaction_class, xyzs=ts_guess_xyzs, solvent=mol.solvent, charge=mol.charge,
                   mult=mol.mult, active_bonds=active_bonds)


def find_1dpes_maximum_energy_xyzs(dist_xyzs_energies_dict):
    """
    Given a 1D list of energies find the maximum that between the end points
    :param dist_xyzs_energies_dict: (dict) [value] = (xyzs, energy)
    :return:
    """

    logger.info('Finding peak in 1D PES')
    xyzs_peak_energy = None
    energy_list = [dist_xyzs_energies_dict[dist][1] for dist in dist_xyzs_energies_dict.keys()]
    peak_e, min_e = min(energy_list), min(energy_list)

    for i in range(1, len(dist_xyzs_energies_dict) - 1):
        if energy_list[i] > peak_e and energy_list[i-1] < energy_list[i] > energy_list[i+1]:
            peak_e = energy_list[i]
            xyzs_peak_energy = list(dist_xyzs_energies_dict.values())[i][0]

    if peak_e != min_e:
        from .constants import Constants
        logger.info('Energy at peak in PES at ∆E = {} kcal/mol'.format(Constants.ha2kcalmol * (peak_e - min_e)))
    else:
        logger.warning('Couldn\'t find a peak in the PES')

    return xyzs_peak_energy
