from autode.log import logger
import numpy as np
from autode.bond_rearrangement import get_bond_rearrangs
from autode.substitution import set_complex_xyzs_translated_rotated
from autode.geom import get_kfitted_coords
from autode.bond_lengths import get_avg_bond_length
from autode.reactions import Substitution, Elimination
from autode.bond_lengths import get_avg_bond_length
from autode.complex import ReactantComplex, ProductComplex
from autode import mol_graphs
from autode.transition_states.optts import get_ts
from autode.transition_states.template_ts_guess import get_template_ts_guess
from autode.pes_1d import get_ts_guess_1d
from autode.pes_2d import get_ts_guess_2d
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.transition_states.transition_state import TS


def find_tss(reaction):
    """Find all the possible the transition states of a reaction

    Arguments:
        reaction (list(autode.reaction.Reaction)): Reaction

    Returns:
        list: list of transition state objects
    """
    logger.info('Finding possible transition states')

    reactant, product = ReactantComplex(*reaction.reacs), ProductComplex(*reaction.prods)
    bond_rearrangs = get_bond_rearrangs(reactant, product, name=reaction.name)

    if bond_rearrangs is None:
        logger.error('Could not find a set of forming/breaking bonds')
        return None

    tss = []
    for bond_rearrangement in bond_rearrangs:
        logger.info(f'Locating transition state using active bonds {bond_rearrangement.all}')
        tss += get_tss(reaction, reactant, product, bond_rearrangement)

    if len(tss) > 0:
        logger.info(f'Found *{len(tss)}* transition state(s) that lead to products')
        return tss

    logger.error('Did not find any transition state(s)')
    return None


def get_ts_guess_funcs_and_params(funcs_params, reaction, reactant, product, bond_rearrang):
    """Get the functions (1dscan or 2dscan) and parameters required for the function for a TS scan

    Args:
        funcs_params (list): current funcs and params
        reaction (reaction object): reaction being examined
        reactant (mol obj): reactant complex
        product (mol obj): product complex
        bond_rearrang (bond rearrang obj): bond rearrangement being scanned
        solvent_mol (mol obj): solvent_name mol, if this is not None explicit qmmm will be done

    Returns:
        list: updated funcs and params list
    """

    name = '+'.join([r.name for r in reaction.reacs]) + '--' + '+'.join([p.name for p in reaction.prods])

    if reactant.is_fragment:
        name += 'fragment_'

    lmethod, hmethod = get_lmethod(), get_hmethod()

    if bond_rearrang.n_bbonds >= 1 and bond_rearrang.n_fbonds >= 1:
        for fbond in bond_rearrang.fbonds:
            for bbond in bond_rearrang.bbonds:
                scanned_bonds = [fbond, bbond]
                active_bonds_not_scanned = [bond for bond in bond_rearrang.all if bond not in scanned_bonds]
                scan_name = name + f'_{fbond[0]}-{fbond[1]}_{bbond[0]}-{bbond[1]}'
                fbond_final_dist = get_avg_bond_length(atom_i_label=reactant.atoms[fbond[0]].label, atom_j_label=reactant.atoms[fbond[1]].label)
                bbond_final_dist = reactant.get_distance(atom_i=bbond[0], atom_j=bbond[1]) + get_added_bbond_dist(reaction)

                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 16, scan_name + '_ll2d', reaction.type, lmethod,
                                                       lmethod.keywords.low_opt, fbond_final_dist, bbond_final_dist, active_bonds_not_scanned)))
                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 8, scan_name + '_hl2d', reaction.type, hmethod,
                                                       lmethod.keywords.low_opt, fbond_final_dist, bbond_final_dist, active_bonds_not_scanned)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 0:
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        bbond_final_dist = reactant.get_distance(atom_i=bbond[0], atom_j=bbond[1]) + get_added_bbond_dist(reaction)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_ll1d', reaction.type,
                                               lmethod, lmethod.keywords.low_opt, bbond_final_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d', reaction.type,
                                               hmethod, hmethod.keywords.low_opt, bbond_final_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level',
                                               reaction.type, hmethod, hmethod.keywords.opt, bbond_final_dist)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 1 and reaction.type in (Substitution, Elimination):
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        bbond_final_dist = reactant.get_distance(atom_i=bbond[0], atom_j=bbond[1]) + get_added_bbond_dist(reaction)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 20, scan_name + '_ll1d_bbond', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, bbond_final_dist, [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_bbond', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, bbond_final_dist, [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level_bbond', reaction.type, hmethod,
                                               hmethod.keywords.opt, bbond_final_dist, [bond_rearrang.fbonds[0]])))

    if bond_rearrang.n_bbonds > 0 and bond_rearrang.n_fbonds == 1:
        fbond = bond_rearrang.fbonds[0]
        scan_name = name + f'_{fbond[0]}-{fbond[1]}'
        fbond_final_dist = get_avg_bond_length(atom_i_label=reactant.atoms[fbond[0]].label, atom_j_label=reactant.atoms[fbond[1]].label)

        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 20, scan_name + '_ll1d_fbond', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, fbond_final_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_fbond', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, fbond_final_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_opt_level_fbond', reaction.type, hmethod,
                                               hmethod.keywords.opt, fbond_final_dist, bond_rearrang.bbonds)))

    if bond_rearrang.n_fbonds == 2:
        fbond1, fbond2 = bond_rearrang.fbonds
        scan_name = name + f'_{fbond1[0]}-{fbond1[1]}_{fbond2[0]}-{fbond2[1]}'
        delta_fbond_dist1 = get_avg_bond_length(atom_i_label=reactant.atoms[fbond1[0]].label, atom_j_label=reactant.atoms[fbond1[1]].label)
        delta_fbond_dist2 = get_avg_bond_length(atom_i_label=reactant.atoms[fbond2[0]].label, atom_j_label=reactant.atoms[fbond2[1]].label)
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 16, scan_name + '_ll2d_fbonds', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 8, scan_name + '_hl2d_fbonds', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))

    if bond_rearrang.n_bbonds == 2:
        bbond1, bbond2 = bond_rearrang.bbonds
        bbond1_final_dist = reactant.get_distance(atom_i=bbond1[0], atom_j=bbond2[1]) + get_added_bbond_dist(reaction)
        bbond2_final_dist = reactant.get_distance(atom_i=bbond1[0], atom_j=bbond2[1]) + get_added_bbond_dist(reaction)

        scan_name = name + f'_{bbond1[0]}-{bbond1[1]}_{bbond2[0]}-{bbond2[1]}'
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 16, scan_name + '_ll2d_bbonds', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, bbond1_final_dist, bbond2_final_dist, bond_rearrang.fbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 8, scan_name + '_hl2d_bbonds', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, bbond1_final_dist, bbond2_final_dist, bond_rearrang.fbonds)))

    return funcs_params


def get_nn_coords(species, atom_index, exclued_atoms):
    """Get the nearest neighbour coordinates to """
    return [species.atoms[nn].coord for nn in list(species.graph.neighbors(atom_index)) if nn not in exclued_atoms]


def translate_rotate_reactant(reactant, product, bond_rearrangement, shift_factor):
    """Find the optimum translation/rotation of the reactant that minimises attacking atom nearest neighbour distance"""
    if len(reactant.molecules) < 2:
        logger.info('No translation/rotation is required')
        return

    # Get the attacked atoms
    attacked_atoms = set()
    for atom_index in bond_rearrangement.active_atoms:
        # Attacked atoms are atoms that have a bond made and simultaneously a bond broken
        if atom_index in bond_rearrangement.fatoms and atom_index in bond_rearrangement.batoms:
            attacked_atoms.add(atom_index)

    # Get the coordinates of the nearest neighbours to the attacked atoms in reactant and products
    nn_coords_reactant, nn_coords_product = [], []
    for atom_index in attacked_atoms:
        nn_coords_reactant += get_nn_coords(reactant, atom_index, exclued_atoms=bond_rearrangement.active_atoms)
        nn_coords_product += get_nn_coords(product, atom_index, exclued_atoms=bond_rearrangement.active_atoms)

    assert len(nn_coords_product) == len(nn_coords_reactant)

    # Apply the rotation to the attacking molecule
    attacking_mol_index = 1 if all([i < reactant.molecules[0].n_atoms for i in attacked_atoms]) else 0
    attacking_mol_indexes = reactant.get_atom_indexes(mol_index=attacking_mol_index)

    # Get the optimum rotation matrix and translation using the Kabash algorithm
    rot_mat, p_trans, q_trans = get_kfitted_coords(template_coords=nn_coords_product, coords_to_fit=nn_coords_reactant)

    # Get all the coordinates of the product
    reac_coords = reactant.get_coordinates()
    prod_coords = product.get_coordinates()

    for atom_index in reactant.get_atom_indexes(mol_index=1 if attacking_mol_index == 0 else 0):
        reac_coords[atom_index] = np.matmul(rot_mat, reac_coords[atom_index] - p_trans) + q_trans

    for atom_index in attacking_mol_indexes:
        reac_coords[atom_index] = prod_coords[atom_index]

    # Shift the attacked mol by the average fbond vector
    fbond_vectors = [product.atoms[fbond[0]].coord - product.atoms[fbond[1]].coord for fbond in bond_rearrangement.bbonds]
    avg_fbond_vector = np.average(np.array(fbond_vectors), axis=0)

    for atom_index in attacking_mol_indexes:
        reac_coords[atom_index] += shift_factor * (avg_fbond_vector / np.linalg.norm(avg_fbond_vector))

    reactant.set_coordinates(reac_coords)
    reactant.print_xyz_file(filename='reactant.xyz')

    exit()
    return None


def get_tss(reaction, reactant, product, bond_rearrangement, strip_molecule=True):
    """For a bond rearrangement, run 1d and 2d scans to find a TS

    Args:
        reaction (reaction object): reaction being examined
        reactant (mol obj): reactant complex
        product (mol obj): product complex
        bond_rearrangement (bond rearrang boj): bond rearrangement being used
        strip_molecule (bool, optional): If true then the molecule will try and be stripped to make the scan
                                         calculations faster. The whole TS can the be found from the template made.
                                         Defaults to True.
    Returns:
        ts obj: the TS of the reaction
    """
    tss = []

    reactant.active_atoms = sorted(set([active_atom for active_atom in bond_rearrangement.active_atoms]))

    # Reorder atoms in product to match reactant
    full_prod_graph_reac_indices = mol_graphs.reac_graph_to_prods(reactant.graph, bond_rearrangement)
    full_mapping = mol_graphs.get_mapping(full_prod_graph_reac_indices, product.graph)

    product.atoms = [product.atoms[i] for i in full_mapping.values()]
    product.graph = mol_graphs.reorder_nodes(graph=product.graph, mapping=full_mapping)

    # If the reaction is a substitution or elimination then the reactants must be orientated correctly
    translate_rotate_reactant(reactant, product, bond_rearrangement,
                              shift_factor=3 if any(mol.charge != 0 for mol in reaction.reacs) else 2)

    exit()
    # if strip_molecule:
    #     reactant_core_atoms = reactant.get_core_atoms(full_prod_graph_reac_indices)
    # else:
    #     reactant_core_atoms = None
#
    # reac_mol, reac_mol_rearrangement = reactant.strip_core(reactant_core_atoms, bond_rearrangement)
#
    # if reac_mol.is_fragment:
    #     product_core_atoms = []
    #     for atom in reactant_core_atoms:
    #         product_core_atoms.append(inv_full_mapping[atom])
    # else:
    #     product_core_atoms = None
#
    # prod_mol, _ = product.strip_core(product_core_atoms)

    funcs_params = [(get_template_ts_guess, (reactant, bond_rearrangement.all, reaction.type, product))]

    for func, params in get_ts_guess_funcs_and_params(funcs_params, reaction, reactant, product, bond_rearrangement):
        logger.info(f'Trying to find a TS guess with {func.__name__}')
        ts_guess = func(*params)

        if ts_guess is None:
            continue

        ts_mol, converged = get_ts(ts_guess)

        if ts_mol is not None:
            ts = TS(ts_mol, converged=converged)
            if ts.is_true_ts():
                logger.info(f'Found a transition state with {func.__name__}')

                # if reac_mol.is_fragment:
                #     logger.info('Finding full TS')
                #     full_ts_guess = get_template_ts_guess(reactant, bond_rearrangement.all, reaction.type, product)
                #     full_ts_mol, full_converged = get_ts(full_ts_guess)
                #     if full_ts_mol is not None:
                #         full_ts = TS(full_ts_mol, converged=full_converged)
                #         if full_ts.is_true_ts():
                #             logger.info('Found full TS')
                #             tss.append(full_ts)
                # else:

                tss.append(ts)
                break

    # if len(tss) == 0 and reac_mol.is_fragment:
    #     logger.info('Found no transition states using the fragment, will try with the whole molecule')
    #     tss = get_tss(reaction, reactant, product, bond_rearrangement, strip_molecule=False)

    return tss


def get_added_bbond_dist(reaction):
    """Get the bond length a breaking bond should increase by

    Args:
        reaction (reaction object): reaction being scanned
        solvent_mol (mol object): solvating molecule (can be none for no explicit solvation)

    Returns:
        float: distance the bond should increase by
    """
    # if there are charged molecules and implicit solvation, the bond length may need to increase by more to get a saddle point,
    # as implicit solvation does not fully stabilise the charged molecule

    # TODO reimplement with explicit solvent like:   if reaction is SolvatedReaction

    if any(mol.charge != 0 for mol in reaction.prods):
        return 2.5
    else:
        return 1.5
