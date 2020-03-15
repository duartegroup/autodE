from autode.log import logger
from autode.bond_rearrangement import get_bond_rearrangs
from autode.substitution import set_complex_xyzs_translated_rotated
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
    bond_rearrangs = get_bond_rearrangs(reactant, product)

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


def get_ts_guess_funcs_and_params(funcs_params, reaction, reactant, product, bond_rearrang, solvent_mol):
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
                active_bonds_not_scanned = [bond for bond in bond_rearrang.all if not bond in scanned_bonds]
                scan_name = name + f'_{fbond[0]}-{fbond[1]}_{bbond[0]}-{bbond[1]}'
                delta_fbond_dist = get_avg_bond_length(mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)
                delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)

                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 16, scan_name + '_ll2d', reaction.type, lmethod,
                                                       lmethod.keywords.low_opt, solvent_mol, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))
                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 8, scan_name + '_hl2d', reaction.type, hmethod,
                                                       lmethod.keywords.low_opt, solvent_mol, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 0:
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_ll1d', reaction.type,
                                               lmethod, lmethod.keywords.low_opt, solvent_mol, delta_bbond_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d', reaction.type,
                                               hmethod, hmethod.keywords.low_opt, solvent_mol, delta_bbond_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level',
                                               reaction.type, hmethod, hmethod.keywords.opt, solvent_mol, delta_bbond_dist)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 1 and reaction.type in (Substitution, Elimination):
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 20, scan_name + '_ll1d_bbond', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, solvent_mol, delta_bbond_dist, [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_bbond', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, solvent_mol, delta_bbond_dist, [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level_bbond', reaction.type, hmethod,
                                               hmethod.keywords.opt, solvent_mol, delta_bbond_dist, [bond_rearrang.fbonds[0]])))

    if bond_rearrang.n_bbonds > 0 and bond_rearrang.n_fbonds == 1:
        fbond = bond_rearrang.fbonds[0]
        scan_name = name + f'_{fbond[0]}-{fbond[1]}'
        delta_fbond_dist = get_avg_bond_length(mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)

        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 20, scan_name + '_ll1d_fbond', reaction.type, lmethod,
                                               lmethod.keywords.low_opt,  solvent_mol, delta_fbond_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_fbond', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, solvent_mol, delta_fbond_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_opt_level_fbond', reaction.type, hmethod,
                                               hmethod.keywords.opt, solvent_mol, delta_fbond_dist, bond_rearrang.bbonds)))

    if bond_rearrang.n_fbonds == 2:
        fbond1, fbond2 = bond_rearrang.fbonds
        scan_name = name + f'_{fbond1[0]}-{fbond1[1]}_{fbond2[0]}-{fbond2[1]}'
        delta_fbond_dist1 = get_avg_bond_length(mol=reactant, bond=fbond1) - reactant.calc_bond_distance(fbond1)
        delta_fbond_dist2 = get_avg_bond_length(mol=reactant, bond=fbond2) - reactant.calc_bond_distance(fbond2)
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 16, scan_name + '_ll2d_fbonds', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, solvent_mol, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 8, scan_name + '_hl2d_fbonds', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, solvent_mol, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))

    if bond_rearrang.n_bbonds == 2:
        bbond1, bbond2 = bond_rearrang.bbonds
        delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)
        scan_name = name + f'_{bbond1[0]}-{bbond1[1]}_{bbond2[0]}-{bbond2[1]}'
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 16, scan_name + '_ll2d_bbonds', reaction.type, lmethod,
                                               lmethod.keywords.low_opt, solvent_mol, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 8, scan_name + '_hl2d_bbonds', reaction.type, hmethod,
                                               hmethod.keywords.low_opt, solvent_mol, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))

    return funcs_params


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

    active_atoms = set()
    for active_atom in bond_rearrangement.active_atoms:
        active_atoms.add(active_atom)

    reactant.active_atoms = sorted(active_atoms)

    # get the product graph with the atom indices of the reactant
    full_prod_graph_reac_indices = mol_graphs.reac_graph_to_prods(reactant.graph, bond_rearrangement)
    # mapping[product indices] = reactant indices
    logger.info('Getting mapping from reactants to products')
    full_mapping = mol_graphs.get_mapping(product.graph, full_prod_graph_reac_indices)
    inv_full_mapping = {v: k for k, v in full_mapping.items()}

    if reaction.type in [Substitution, Elimination]:
        if any(mol.charge != 0 for mol in reaction.reacs):
            shift_factor = 3
        else:
            shift_factor = 2
        set_complex_xyzs_translated_rotated(reactant, product, reaction.reacs, bond_rearrangement, inv_full_mapping, shift_factor)

    if strip_molecule:
        reactant_core_atoms = reactant.get_core_atoms(full_prod_graph_reac_indices)
    else:
        reactant_core_atoms = None

    reac_mol, reac_mol_rearrangement = reactant.strip_core(reactant_core_atoms, bond_rearrangement)

    if reac_mol.is_fragment:
        product_core_atoms = []
        for atom in reactant_core_atoms:
            product_core_atoms.append(inv_full_mapping[atom])
    else:
        product_core_atoms = None

    prod_mol, _ = product.strip_core(product_core_atoms)

    funcs_params = [(get_template_ts_guess, (reactant, bond_rearrangement.all, reaction.type, product))]

    for func, params in get_ts_guess_funcs_and_params(funcs_params, reaction, reac_mol, prod_mol, reac_mol_rearrangement, solvent_mol):
        logger.info(f'Trying to find a TS guess with {func.__name__}')
        ts_guess = func(*params)

        if ts_guess is None:
            continue

        ts_mol, converged = get_ts(ts_guess, solvent_mol)

        if ts_mol is not None:
            ts = TS(ts_mol, converged=converged)
            if ts.is_true_ts():
                logger.info(f'Found a transition state with {func.__name__}')
                if reac_mol.is_fragment:
                    logger.info('Finding full TS')
                    full_ts_guess = get_template_ts_guess(reactant, bond_rearrangement.all, reaction.type, product)
                    full_ts_mol, full_converged = get_ts(full_ts_guess)
                    if full_ts_mol is not None:
                        full_ts = TS(full_ts_mol, converged=full_converged)
                        if full_ts.is_true_ts():
                            logger.info('Found full TS')
                            tss.append(full_ts)
                else:
                    tss.append(ts)
                break

    if len(tss) == 0 and reac_mol.is_fragment:
        logger.info('Found no transition states using the fragment, will try with the whole molecule')
        tss = get_ts(reaction, reactant, product, bond_rearrangement, solvent_mol, strip_molecule=False)

    return tss


def get_bbond_dist(reaction, solvent_mol):
    """Get the bond length a breaking bond should increase by

    Args:
        reaction (reaction object): reaction being scanned
        solvent_mol (mol object): solvating molecule (can be none for no explicit solvation)

    Returns:
        float: distance the bond should increase by
    """
    # if there are charged molecules and implicit solvation, the bond length may need to increase by more to get a saddle point,
    # as implicit solvation does not fully stabilise the charged molecule
    if any(mol.charge != 0 for mol in reaction.prods) and solvent_mol is None:
        return 2.5
    else:
        return 1.5
