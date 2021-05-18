import numpy as np
from scipy.optimize import minimize
from autode.exceptions import NoMapping
from autode.atoms import metals
from autode.transition_states.transition_state import get_ts_object
from autode.transition_states.truncation import get_truncated_complex
from autode.transition_states.truncation import is_worth_truncating
from autode.transition_states.ts_guess import get_template_ts_guess
from autode.bond_rearrangement import get_bond_rearrangs
from autode.config import Config
from autode.log import logger
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.mol_graphs import get_mapping
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.mol_graphs import reorder_nodes
from autode.bonds import FormingBond, BreakingBond
from autode.path.adaptive import get_ts_adaptive_path
from autode.mol_graphs import species_are_isomorphic
from autode.substitution import get_cost_rotate_translate
from autode.substitution import get_substitution_centres


def find_tss(reaction):
    """
    Find all the possible the transition states of a reaction over possible
    paths from reaction.reactant to reaction.product. Will not search the
    conformational space of a reaction

    Arguments:
        (list(autode.reaction.Reaction)): Reaction

    Returns:
        (list(autode.transition_states.transition_state.TransitionState)):
    """
    logger.info('Finding possible transition states')
    reactant, product = reaction.reactant, reaction.product

    if reactant is None or product is None:
        raise ValueError('Reaction must have reaction.reactant and'
                         ' reaction.product set as species')

    if species_are_isomorphic(reactant, product):
        logger.error('Reactant and product complexes are isomorphic. Cannot'
                     ' find a TS')
        return None

    bond_rearrs = get_bond_rearrangs(reactant, product, name=str(reaction))

    if bond_rearrs is None:
        logger.error('Could not find a set of forming/breaking bonds')
        return None

    tss = []
    for bond_rearrangement in bond_rearrs:
        logger.info(f'Locating transition state using active bonds '
                    f'{bond_rearrangement.all}')

        ts = get_ts(reaction, reactant, bond_rearrangement)

        if ts is not None:
            tss.append(ts)

    if len(tss) == 0:
        logger.error('Did not find any transition state(s)')
        return None

    logger.info(f'Found *{len(tss)}* transition state(s) that lead to products')
    return tss


def get_ts_guess_function_and_params(reaction, bond_rearr):
    """Get the functions (1dscan or 2dscan) and parameters required for the
    function for a TS scan

    Arguments:
        reaction (autode.reaction.Reaction):
        bond_rearr (autode.bond_rearrangement.BondRearrangement):

    Yields:
        (tuple(func, args)):
    """
    name = str(reaction)
    scan_name = name

    r, p = reaction.reactant, reaction.product

    lmethod, hmethod = get_lmethod(), get_hmethod()

    # Bonds with initial and final distances
    bbonds = [BreakingBond(pair, r, p) for pair in bond_rearr.bbonds]
    scan_name += "_".join(str(bb) for bb in bbonds)

    fbonds = [FormingBond(pair, r, p) for pair in bond_rearr.fbonds]
    scan_name += "_".join(str(fb) for fb in fbonds)

    # Ideally use a transition state template, then only a single constrained
    # optimisation needs to be run
    yield get_template_ts_guess, (r, p, bond_rearr,
                                  f'{name}_template_{bond_rearr}', hmethod)

    # otherwise try a nudged elastic band calculation, don't use the low level
    # method if there are any metals
    if not any(atom.label in metals for atom in r.atoms):
        yield get_ts_adaptive_path, (r, p, lmethod, fbonds, bbonds,
                                     f'{name}_ll_ad_{bond_rearr}')

    # Always attempt a high-level NEB
    yield get_ts_adaptive_path, (r, p, hmethod, fbonds, bbonds,
                                 f'{name}_hl_ad_{bond_rearr}')

    return None


def translate_rotate_reactant(reactant, bond_rearrangement, shift_factor,
                              n_iters=10):
    """
    Shift a molecule in the reactant complex so that the attacking atoms
    (a_atoms) are pointing towards the attacked atoms (l_atoms). Applied in
    place

    Arguments:
        reactant (autode.complex.ReactantComplex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):

        shift_factor (float):

        n_iters (int): Number of iterations of translation/rotation to perform
                       to (hopefully) find the global minima
    """
    if not hasattr(reactant, 'molecules'):
        logger.warning('Cannot rotate/translate component, not a Complex')
        return

    if len(reactant.molecules) < 2:
        logger.info('Reactant molecule does not need to be translated or '
                    'rotated')
        return

    logger.info('Rotating/translating into a reactive conformation... running')

    # This function can add dummy atoms for e.g. SN2' reactions where there
    # is not a A -- C -- Xattern for the substitution centre
    subst_centres = get_substitution_centres(reactant,
                                             bond_rearrangement,
                                             shift_factor=shift_factor)

    if all(sc.a_atom in reactant.get_atom_indexes(mol_index=0)
           for sc in subst_centres):
        attacking_mol = 0
    else:
        attacking_mol = 1

    # Disable the logger to prevent rotation/translations printing
    logger.disabled = True

    # Find the global minimum for inplace rotation, translation and rotation
    min_cost, opt_x = None, None

    for _ in range(n_iters):
        res = minimize(get_cost_rotate_translate,
                       x0=np.random.random(11),
                       method='BFGS',
                       tol=0.1,
                       args=(reactant, subst_centres, attacking_mol))

        if min_cost is None or res.fun < min_cost:
            min_cost = res.fun
            opt_x = res.x

    # Renable the logger
    logger.disabled = False
    logger.info(f'Minimum cost for translating/rotating is {min_cost:.3f}')

    # Translate/rotation the attacking molecule optimally
    reactant.rotate_mol(axis=opt_x[:3], theta=opt_x[3], mol_index=attacking_mol)
    reactant.translate_mol(vec=opt_x[4:7], mol_index=attacking_mol)
    reactant.rotate_mol(axis=opt_x[7:10], theta=opt_x[10], mol_index=attacking_mol)

    logger.info('                                                 ... done')
    reactant.print_xyz_file()

    # Remove any dummy atoms that may have been added
    # in alt_substitution_centres
    reactant.atoms = [atom for atom in reactant.atoms if atom.label != 'D']

    return None


def get_truncated_ts(reaction, bond_rearr):
    """Get the TS of a truncated reactant and product complex"""

    # Truncate the reactant and product complex to the core atoms so the full
    # TS can be template-d
    f_reactant = reaction.reactant.copy()
    f_product = reaction.product.copy()

    # Set the truncated reactant and product for this reaction
    reaction.reactant = get_truncated_complex(f_reactant, bond_rearr)
    reaction.product = get_truncated_complex(f_product, bond_rearr)

    # Re-find the bond rearrangements, which should exist
    reaction.name += '_truncated'
    bond_rearrangs = get_bond_rearrangs(reaction.reactant, reaction.product,
                                        name=reaction.name)

    if bond_rearrangs is None:
        logger.error('Truncation generated a complex with 0 rearrangements')
        return None

    # Find all the possible TSs
    for bond_rearr in bond_rearrangs:
        get_ts(reaction, reaction.reactant, bond_rearr,  is_truncated=True)

    # Reset the reactant, product and name of the full reaction
    reaction.reactant = f_reactant
    reaction.product = f_product
    reaction.name = reaction.name.rstrip('_truncated')

    logger.info('Done with truncation')
    return None


def reorder_product(reactant, product, bond_rearr):
    """
    Reorder the atoms in the product, and its molecular graph to reflect those
    in the reactant.

    NOTE: This will apply the first valid atom mapping which
    is closest to the reactant, not necessarily the 'true' mapping

    Arguments:
        reactant (autode.complex.ReactantComplex):

        product (autode.complex.ProductComplex):

        bond_rearr (autode.bond_rearrangement.BondRearrangement):
    """
    reordered_product = product.copy()

    mapping = get_mapping(graph1=reordered_product.graph,
                          graph2=reac_graph_to_prod_graph(reactant.graph, bond_rearr))

    reordered_product.atoms = [reordered_product.atoms[i] for i in sorted(mapping, key=mapping.get)]

    reordered_product.graph = reorder_nodes(graph=reordered_product.graph,
                                            mapping={u: v for v, u in mapping.items()})
    return reordered_product


def get_ts(reaction, reactant, bond_rearr, is_truncated=False):
    """For a bond rearrangement run PES exploration and TS optimisation to
    find a TS

    Arguments:
        reaction (autode.reaction.Reaction):
        reactant (autode.complex.ReactantComplex):
        bond_rearr (autode.bond_rearrangement.BondRearrangement):
        is_truncated (bool, optional): If the reactant is already truncated
                                       then truncation shouldn't be attempted
                                       and there should be no need to shift
    Returns:
        (autode.transition_states.transition_state.TransitionState): TS
    """
    if reaction.product is None or reaction.reactant is None:
        logger.warning('Reaction had no complexes - generating')
        reaction.find_complexes()

    if bond_rearr.n_fbonds > bond_rearr.n_bbonds:
        raise NotImplementedError('Cannot treat more forming than breaking '
                                  'bonds, reverse the reaction(?)')

    # Reorder the atoms in the product complex so they are equivalent to the
    # reactant
    try:
        reaction.product = reorder_product(reactant,
                                           reaction.product,
                                           bond_rearr)
    except NoMapping:
        logger.warning('Could not find the expected bijection R -> P')
        return None

    # If the reaction is a substitution or elimination then the reactants must
    # be orientated correctly, no need to re-rotate/translate if truncated
    if not is_truncated:
        translate_rotate_reactant(reactant, bond_rearrangement=bond_rearr,
                                  shift_factor=1.5 if reactant.charge == 0 else 2.5)

    # If specified then strip non-core atoms from the structure
    if is_worth_truncating(reactant, bond_rearr) and not is_truncated:
        get_truncated_ts(reaction, bond_rearr)

    # There are multiple methods of finding a transition state. Iterate through
    # from the cheapest -> most expensive
    for func, params in get_ts_guess_function_and_params(reaction, bond_rearr):
        logger.info(f'Trying to find a TS guess with {func.__name__}')
        ts_guess = func(*params)

        if ts_guess is None:
            continue

        ts_guess.bond_rearrangement = bond_rearr

        if not ts_guess.could_have_correct_imag_mode():
            continue

        # Form a transition state object and run an OptTS calculation
        ts = get_ts_object(ts_guess)
        ts.optimise()

        if not ts.is_true_ts():
            continue

        # Save a transition state template if specified in the config
        if Config.make_ts_template:
            ts.save_ts_template(folder_path=Config.ts_template_folder_path)

        logger.info(f'Found a transition state with {func.__name__}')
        return ts

    return None
