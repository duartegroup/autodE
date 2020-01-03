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
from autode import mol_graphs
from autode.pes_2d import replace_none


def get_ts_guess_1d(mol, product, active_bond, n_steps, name, reaction_class, method, keywords, delta_dist=1.5,
                    active_bonds_not_scanned=None):
    """Scan the distance between two atoms and return a guess for the TS

    Arguments:
        mol {molcule object} -- reactant complex
        product {molecule object} -- product complex
        active_bond {tuple} -- tuple of atom ids showing the bond being scanned
        n_steps {int} -- number of steps to take in the scan
        name {str} -- name of reaction
        reaction_class {object} -- class of the reaction (reactions.py)
        method {object} -- electronic structure wrapper to use for the calcs
        keywords {list} -- keywords to use in the calcs

    Keyword Arguments:
        delta_dist {float} -- distance to add onto the current distance (Å) in n_steps (default: {1.5})
        active_bonds_not_scanned {list(tuple)} -- pairs of atoms that are active, but will not be scanned in the 1D PES (default: {None})

    Returns:
        {ts guess object} -- ts guess
    """
    logger.info(f'Getting TS guess from 1D relaxed potential energy scan using {active_bond} as the active bond')
    mol_with_const = deepcopy(mol)

    curr_dist = mol.calc_bond_distance(active_bond)
    # Generate a list of distances at which to constrain the optimisation
    dists = np.linspace(curr_dist, curr_dist + delta_dist, n_steps)
    # Initialise an empty dictionary containing the distance as a key and the xyzs and energy as s tuple value
    xyzs_list, energy_list = [], []

    # Run a relaxed potential energy surface scan by running sequential constrained optimisations
    for n, dist in enumerate(dists):
        const_opt = Calculation(name=name + f'_scan{n}', molecule=mol_with_const, method=method, opt=True,
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
    expected_prod_graphs = mol_graphs.get_separate_subgraphs(product.graph)
    logger.info('Checking products were made')
    ts_product_graphs = [mol_graphs.make_graph(xyzs, mol.n_atoms)
                         for xyzs in xyzs_list[::-1]]
    products_made = False
    for ts_product_graph in ts_product_graphs:
        if all(mol_graphs.is_subgraph_isomorphic(ts_product_graph, graph) for graph in expected_prod_graphs):
            products_made = True
            logger.info('Products made')
            break

    if not products_made:
        logger.info('Products not made')
        return None

    make_reaction_animation(name, [mol.xyzs] + xyzs_list)

    # Make a new molecule that will form the basis of the TS guess object
    tsguess_mol = deepcopy(mol)
    tsguess_mol.set_xyzs(xyzs=find_1dpes_maximum_energy_xyzs(dists, xyzs_list, energy_list, scan_name=name, plot_name=mol.name + f'_{active_bond[0]}_{active_bond[1]}_1dscan', method=method))

    if tsguess_mol.xyzs is None:
        logger.warning('TS guess had no xyzs')
        return None

    active_bonds = [active_bond] if active_bonds_not_scanned is None else [active_bond] + active_bonds_not_scanned

    return TSguess(name=name, reaction_class=reaction_class, molecule=tsguess_mol, active_bonds=active_bonds, reactant=mol, product=product)


def find_1dpes_maximum_energy_xyzs(dist_list, xyzs_list, energy_list, scan_name, plot_name, method):
    """Given a 1D list of energies find the maximum that lies between the end points

    Arguments:
        dist_list {np.ndarray} -- list of distances the scan points are at
        xyzs_list {list(list(list))} -- list of the xyzs at the scan point
        energy_list {list(float)} -- list of energies at the scan points
        scan_name {str} -- name of the reaction
        plot_name {str} -- name of the plot made
        method {object} -- electronic structure wrapper used for the calcs

    Returns:
        {list(list)} -- xyzs at the peak of the PES
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

    plot_1dpes(dist_list, [Constants.ha2kcalmol * (e - min_e)
                           for e in energies_not_none], scan_name=scan_name, plot_name=plot_name, method=method)

    if peak_e != min_e:
        logger.info(f'Energy at peak in PES at ∆E = {Constants.ha2kcalmol * (peak_e - min_e)} kcal/mol')
    else:
        logger.warning('Couldn\'t find a peak in the PES')

    return xyzs_peak_energy
