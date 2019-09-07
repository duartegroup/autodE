import numpy as np
from copy import deepcopy
from autode.config import Config
from autode.log import logger
from autode.ts_guess import TSguess
from autode.plotting import plot_1dpes
from autode.constants import Constants
from autode.calculation import Calculation
from autode.wrappers.ORCA import ORCA
from autode.wrappers.XTB import XTB


def get_est_ts_guess_1dpes_scan(mol, active_bond, n_steps, orca_keywords, name, reaction_class, delta_dist=1.5,
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
    curr_dist = mol.calc_bond_distance(active_bond)

    scan = Calculation(name=name + '_scan', molecule=mol, method=ORCA, keywords=orca_keywords,
                       n_cores=Config.n_cores, max_core_mb=Config.max_core, scan_ids=active_bond,
                       curr_dist1=curr_dist, final_dist1=curr_dist + delta_dist,  opt=True, n_steps=n_steps)

    scan.run()
    dist_xyzs_energies = scan.get_scan_values_xyzs_energies()
    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_1dpes_maximum_energy_xyzs(dist_xyzs_energies))

    if tsguess_mol.xyzs is None:
        return None

    active_bonds = [active_bond] if active_bonds_not_scanned is None else [active_bond] + active_bonds_not_scanned

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol, active_bonds=active_bonds)


def get_xtb_ts_guess_1dpes_scan(mol, active_bond, n_steps, name, reaction_class, delta_dist=1.5,
                                active_bonds_not_scanned=None):
    """
    Scan the distance between 2 atoms and return the xyzs with peak energy
    :param mol: Molecule object
    :param active_bond: (tuple) of atom ids
    :param delta_dist: (float) Distance to add onto the current distance (Å)
    :param n_steps: (int) Number of scan steps to use in the XTB scan
    :param name: (str) Name of reaction
    :param reaction_class: (object) class of the reaction (reactions.py)
    :param active_bonds_not_scanned: list(tuple) pairs of atoms that are active, but will not be scanned in the 1D PES
    :return: List of xyzs
    """
    logger.info('Getting TS guess from XTB relaxed potential energy scan')

    curr_dist = mol.calc_bond_distance(active_bond)
    dists = np.linspace(curr_dist, curr_dist + delta_dist, n_steps)
    mol_with_const = deepcopy(mol)
    dist_xyzs_energies = {}

    # Run a relaxed potential energy surface scan using XTB by running sequential constrained optimisations
    for n, dist in enumerate(dists):
        const_opt = Calculation(name=name + '_scan' + str(n), molecule=mol_with_const, method=XTB, opt=True,
                                n_cores=Config.n_cores, distance_constraints={active_bond: dist})
        const_opt.run()
        xyzs = const_opt.get_final_xyzs()
        dist_xyzs_energies[dist] = (xyzs, const_opt.get_energy())
        mol_with_const.xyzs = xyzs

    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_1dpes_maximum_energy_xyzs(dist_xyzs_energies))

    if tsguess_mol.xyzs is None:
        return None

    active_bonds = [active_bond] if active_bonds_not_scanned is None else [active_bond] + active_bonds_not_scanned

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol, active_bonds=active_bonds)


def find_1dpes_maximum_energy_xyzs(dist_xyzs_energies_dict):
    """
    Given a 1D list of energies find the maximum that between the end points
    :param dist_xyzs_energies_dict: (dict) [value] = (xyzs, energy)
    :return:
    """

    logger.info('Finding peak in 1D PES')
    xyzs_peak_energy = None
    if dist_xyzs_energies_dict is None:
        logger.error('Had no distances, xyzs and energies')
        return None

    energy_list = [dist_xyzs_energies_dict[dist][1] for dist in dist_xyzs_energies_dict.keys()]
    peak_e, min_e = min(energy_list), min(energy_list)

    for i in range(1, len(dist_xyzs_energies_dict) - 1):
        if energy_list[i] > peak_e and energy_list[i-1] < energy_list[i] > energy_list[i+1]:
            peak_e = energy_list[i]
            xyzs_peak_energy = list(dist_xyzs_energies_dict.values())[i][0]

    plot_1dpes(dist_xyzs_energies_dict.keys(), [Constants.ha2kcalmol * (e - min_e) for e in energy_list])

    if peak_e != min_e:
        logger.info('Energy at peak in PES at ∆E = {} kcal/mol'.format(Constants.ha2kcalmol * (peak_e - min_e)))
    else:
        logger.warning('Couldn\'t find a peak in the PES')

    return xyzs_peak_energy
