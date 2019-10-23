import numpy as np
from copy import deepcopy
from autode.config import Config
from autode.log import logger
from autode.transition_states.ts_guess import TSguess
from autode.plotting import plot_1dpes
from autode.plotting import make_reaction_animation
from autode.constants import Constants
from autode.calculation import Calculation
from autode.exceptions import XYZsNotFound
from autode.mol_graphs import make_graph
from autode.mol_graphs import is_subgraph_isomorphic
from autode.pes_2d import replace_none


def get_ts_guess_1dpes_scan(mol, product, active_bond, n_steps, name, reaction_class, method, keywords, products, delta_dist=1.5,
                            active_bonds_not_scanned=None):
    """
    Scan the distance between 2 atoms and return the xyzs with peak energy
    :param mol: Molecule object
    :param product: single molecule object of the products
    :param active_bond: (tuple) of atom ids
    :param method: (object) electronic structure method
    :param keywords (list) list of keywords required by an electronic structure method
    :param delta_dist: (float) Distance to add onto the current distance (Å)
    :param n_steps: (int) Number of scan steps to use in the XTB scan
    :param name: (str) Name of reaction
    :param reaction_class: (object) class of the reaction (reactions.py)
    :param products: (object) list of product molecule objects
    :param active_bonds_not_scanned: list(tuple) pairs of atoms that are active, but will not be scanned in the 1D PES
    :return: List of xyzs
    """
    logger.info(
        f'Getting TS guess from 1D relaxed potential energy scan using {active_bond} as the active bond')
    mol_with_const = deepcopy(mol)

    curr_dist = mol.calc_bond_distance(active_bond)
    # Generate a list of distances at which to constrain the optimisation
    dists = np.linspace(curr_dist, curr_dist + delta_dist, n_steps)
    # Initialise an empty dictionary containing the distance as a key and the xyzs and energy as s tuple value
    xyzs_list, energy_list = [], []

    # Run a relaxed potential energy surface scan by running sequential constrained optimisations
    for n, dist in enumerate(dists):
        const_opt = Calculation(name=name + '_scan' + str(n), molecule=mol_with_const, method=method, opt=True,
                                n_cores=Config.n_cores, distance_constraints={active_bond: dist}, keywords=keywords)
        const_opt.run()

        # Set the new xyzs as those output from the calculation, and the previous if no xyzs could be found
        try:
            xyzs = const_opt.get_final_xyzs()
        except XYZsNotFound:
            logger.error('Could not find XYZs, setting as previous')
            xyzs = deepcopy(const_opt.xyzs)

        xyzs_list.append(xyzs)
        energy_list.append(const_opt.get_energy())

        # Update the molecule with constraints xyzs such that the next optimisation is as fast as possible
        mol_with_const.xyzs = xyzs

    # check product and TSGuess product graphs are isomorphic
    logger.info('Checking products were made')
    ts_product_graphs = [make_graph(xyzs, mol.n_atoms)
                         for xyzs in xyzs_list[::-1]]
    products_made = False
    for graph in ts_product_graphs:
        if all(is_subgraph_isomorphic(graph, product.graph) for product in products):
            products_made = True
        if products_made:
            break

    if not products_made:
        logger.info('Products not made')
        return None

    # Make a new molecule that will form the basis of the TS guess object
    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_1dpes_maximum_energy_xyzs(
        dists, xyzs_list, energy_list, scan_name=name, plot_name=mol.name + '_1dscan', method=method))

    if tsguess_mol.xyzs is None:
        logger.warning('TS guess had no xyzs')
        return None

    make_reaction_animation(name, [mol.xyzs] + xyzs_list)

    active_bonds = [active_bond] if active_bonds_not_scanned is None else [
        active_bond] + active_bonds_not_scanned

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol, active_bonds=active_bonds, reactant=mol, product=product)


def find_1dpes_maximum_energy_xyzs(dist_list, xyzs_list, energy_list, scan_name, plot_name, method):
    """
    Given a 1D list of energies find the maximum that between the end points
    :param dist_list: (ndarray)
    :param xyzs_list: (list)
    :param energy_list: (list)
    :param scan_name: (str)
    :param plot_name: (str)
    :return:
    """

    logger.info('Finding peak in 1D PES')

    xyzs_peak_energy = None

    energies_not_none = list(replace_none(energy_list))

    if len(xyzs_list) == 0 or len(energy_list) == 0:
        logger.error('Had no distances, xyzs and energies')
        return None

    peak_e, min_e = min(energies_not_none), min(energies_not_none)

    for i in range(1, len(dist_list) - 1):
        if energies_not_none[i] > peak_e and energies_not_none[i-1] <= energies_not_none[i] >= energies_not_none[i+1]:
            peak_e = energy_list[i]
            xyzs_peak_energy = xyzs_list[i]

    logger.info('Plotting 1D scan and saving to {}.png'.format(plot_name))
    plot_1dpes(dist_list, [Constants.ha2kcalmol * (e - min_e)
                           for e in energies_not_none], scan_name=scan_name, plot_name=plot_name, method=method)

    if peak_e != min_e:
        logger.info(
            'Energy at peak in PES at ∆E = {} kcal/mol'.format(Constants.ha2kcalmol * (peak_e - min_e)))
    else:
        logger.warning('Couldn\'t find a peak in the PES')

    return xyzs_peak_energy
