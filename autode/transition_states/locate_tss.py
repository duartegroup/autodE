from autode.log import logger
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from autode.geom import length
from autode.bond_rearrangement import get_bond_rearrangs
from autode.substitution import get_substitution_centres
from autode.reactions import Substitution, Elimination
from autode.bond_lengths import get_avg_bond_length
from autode.complex import ReactantComplex, ProductComplex
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


def attack_cost(reactant, subst_centres, attacking_mol_idx, a=1.0, b=1.0, c=0.5, d=10.0):
    """
    Calculate the 'attack cost' for a molecule attacking in e.g. a substitution or elimination reaction.

    C = Σ_ac a * (r_ac - r^0_ac)^2  +  Σ_acx b * (1 - cos(θ))  +  Σ_acx c*(1 + cos(φ))  +  Σ_ij d/r_ij^4

    where cos(θ) = (v_ann • v_cx / |v_ann||v_cx|)
          cos(φ) = (v_ca • v_cx / |v_ca||v_cx|)


    """
    coords = reactant.get_coordinates()
    cost = 0

    for subst_centre in subst_centres:

        r_ac = reactant.get_distance(atom_i=subst_centre.a_atom, atom_j=subst_centre.c_atom)
        cost += a * (r_ac - subst_centre.r0_ac)**2

        # Attack vector is the average of all the nearest neighbour atoms, unless it is flat
        a_nn_coords = [coords[atom_index] - coords[subst_centre.a_atom] for atom_index in subst_centre.a_atom_nn]
        v_ann = -np.average(np.array(a_nn_coords), axis=0)

        # TODO is this a reasonable length?
        if length(v_ann) < 1E-4:
            logger.info('Attacking atom is planar. Computing the perpendicular from two nearest neighbours')
            assert len(subst_centre.a_atom_nn) > 1
            v_ann = np.cross(coords[subst_centre.a_atom] - coords[subst_centre.a_atom_nn[0]],
                             coords[subst_centre.a_atom] - coords[subst_centre.a_atom_nn[1]])

        v_cx = coords[subst_centre.x_atom] - coords[subst_centre.c_atom]
        cost += b * (1 - np.dot(v_ann, v_cx) / (length(v_ann) * length(v_cx)))      # b(1 - cos(θ))

        v_ca = coords[subst_centre.c_atom] - coords[subst_centre.a_atom]
        cost += c * (1 + np.dot(v_ca, v_cx) / (length(v_ca) * length(v_cx)))        # b(1 - cos(φ))

        repulsion = reactant.calc_repulsion(mol_index=attacking_mol_idx)
        cost += d * repulsion

    return cost


def get_cost_rotate_translate(x, reactant, subst_centres, attacking_mol_idx):
    """
    Get the cost for placing an attacking mol given a specified rotation and translation

    Arguments:
        x (np.ndarray): Length 11
        reactant (autode.complex.ReactantComplex):
        subst_centres (list(autode.substitution.SubstitutionCentre)):
        attacking_mol_idx (int): Index of the attacking molecule

    """

    moved_reactant = deepcopy(reactant)
    moved_reactant.rotate_mol(axis=x[:3], theta=x[3], mol_index=attacking_mol_idx)
    moved_reactant.translate_mol(vec=x[4:7], mol_index=attacking_mol_idx)
    moved_reactant.rotate_mol(axis=x[7:10], theta=x[10], mol_index=attacking_mol_idx)

    return attack_cost(moved_reactant, subst_centres, attacking_mol_idx)


def translate_rotate_reactant(reactant, bond_rearrangement, shift_factor):
    """
    Shift a molecule in the reactant complex so that the attacking atoms (a_atoms) are pointing towards the
    attacked atoms (l_atoms)

    Arguments:
        reactant (autode.complex.Complex):
        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
        shift_factor (float):
    """
    if len(reactant.molecules) < 2:
        logger.info('Reactant molecule does not need to be translated or rotated')
        return

    logger.info('Rotating/translating the attacking molecule into a reactive conformation')

    subst_centres = get_substitution_centres(reactant, bond_rearrangement, shift_factor=shift_factor)
    attacking_mol = 0 if all(sc.a_atom in reactant.get_atom_indexes(mol_index=0) for sc in subst_centres) else 1

    logger.disabled = True
    res = minimize(get_cost_rotate_translate, x0=np.random.random(11), method='BFGS', tol=0.1,
                   args=(reactant, subst_centres, attacking_mol))
    logger.disabled = False

    reactant.translate_mol(vec=res.x[:3], mol_index=attacking_mol)
    reactant.rotate_mol(axis=res.x[3:6], theta=res.x[6], mol_index=attacking_mol)

    reactant.print_xyz_file(filename='reactant.xyz')

    print(res.fun)
    # cost = attack_cost(reactant, subst_centres=subst_centres, attacking_mol_idx=attacking_mol)
    # print(cost)




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

    active_atoms = set([active_atom for active_atom in bond_rearrangement.active_atoms])
    reactant.active_atoms = sorted(active_atoms)

    # If the reaction is a substitution or elimination then the reactants must be orientated correctly
    translate_rotate_reactant(reactant, bond_rearrangement,
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
#
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
