import os
import numpy as np
from scipy.optimize import minimize
from autode.exceptions import NoMapping
from autode.species import Complex
from autode.transition_states import TransitionState, TransitionStates
from autode.transition_states.truncation import get_truncated_species
from autode.transition_states.truncation import is_worth_truncating
from autode.transition_states.ts_guess import get_template_ts_guess
from autode.bond_rearrangement import get_bond_rearrangs
from autode.config import Config
from autode.log import logger
from autode.values import Distance, PotentialEnergy
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.utils import work_in
from autode.mol_graphs import get_mapping
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.bonds import FormingBond, BreakingBond
from autode.path.adaptive import get_ts_adaptive_path
from autode.mol_graphs import species_are_isomorphic
from autode.substitution import get_cost_rotate_translate
from autode.substitution import get_substc_and_add_dummy_atoms


def find_tss(reaction):
    """
    Find all the possible the transition states of a reaction over possible
    paths from reaction.reactant to reaction.product. Will not search the
    conformational space of a reaction

    ---------------------------------------------------------------------------
    Arguments:
        (list(autode.reaction.Reaction)): Reaction

    Returns:
        (autode.transition_states.transition_states.TransitionStates):
    """
    logger.info("Finding possible transition states")
    reactant, product = reaction.reactant, reaction.product

    if species_are_isomorphic(reactant, product):
        raise ValueError(
            "Reactant and product complexes are isomorphic. "
            "Cannot find a TS"
        )

    bond_rearrs = get_bond_rearrangs(reactant, product, name=str(reaction))

    if bond_rearrs is None:
        logger.error("Could not find a set of forming/breaking bonds")
        return None

    tss = TransitionStates()
    for bond_rearrangement in bond_rearrs:
        logger.info(
            f"Locating transition state using active bonds "
            f"{bond_rearrangement.all}"
        )

        ts = get_ts(str(reaction), reactant, product, bond_rearrangement)

        if ts is not None:
            tss.append(ts)

    logger.info(
        f"Found *{len(tss)}* transition state(s) that lead to products"
    )
    return tss


def ts_guess_funcs_prms(name, reactant, product, bond_rearr):
    """
    Get the functions and parameters required for the function

    ---------------------------------------------------------------------------
    Arguments:
        name (str): Unique identifier for this reaction

        reactant (autode.species.Species):

        product (autode.species.Species):

        bond_rearr (autode.bond_rearrangement.BondRearrangement):

    Yields:
        (tuple(func, args)):
    """
    r, p = reactant.copy(), product.copy()  # Reactants/products may be edited

    lmethod, hmethod = get_lmethod(), get_hmethod()

    # TODO: make this less awful (consistent types)
    for i, pair in enumerate(bond_rearr.bbonds):
        bond_rearr.bbonds[i] = BreakingBond(pair, r, p)

    for i, pair in enumerate(bond_rearr.fbonds):
        bond_rearr.fbonds[i] = FormingBond(pair, r, p)
    # TODO: -------------------------------------------

    # Ideally use a transition state template, then only a single constrained
    # optimisation needs to be run
    yield get_template_ts_guess, (
        r,
        p,
        bond_rearr,
        f"{name}_template_{bond_rearr}",
        hmethod,
    )

    if (not r.atoms.contain_metals) and hmethod != lmethod:
        yield get_ts_adaptive_path, (
            r,
            p,
            lmethod,
            bond_rearr,
            f"{name}_ll_ad_{bond_rearr}",
        )

    yield get_ts_adaptive_path, (
        r,
        p,
        hmethod,
        bond_rearr,
        f"{name}_hl_ad_{bond_rearr}",
    )

    yield _get_ts_neb_from_adaptive_path, (
        r,
        p,
        hmethod,
        bond_rearr,
        f"{name}_hl_ad_neb_{bond_rearr}",
        f"{name}_hl_ad_{bond_rearr}",
    )
    return None


def translate_rotate_reactant(
    reactant, bond_rearrangement, shift_factor, n_iters=10
):
    """
    Shift a molecule in the reactant complex so that the attacking atoms
    (a_atoms) are pointing towards the attacked atoms (l_atoms). Applied in
    place

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.Complex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):

        shift_factor (float):

        n_iters (int): Number of iterations of translation/rotation to perform
                       to (hopefully) find the global minima
    """

    if not isinstance(reactant, Complex):
        logger.warning("Cannot rotate/translate component, not a Complex")
        return

    if reactant.n_molecules < 2:
        logger.info(
            "Reactant molecule does not need to be translated or " "rotated"
        )
        return

    logger.info("Rotating/translating into a reactive conformation... running")

    # This function can add dummy atoms for e.g. SN2' reactions where there
    # is not a A -- C -- Xattern for the substitution centre
    subst_centres = get_substc_and_add_dummy_atoms(
        reactant, bond_rearrangement, shift_factor=shift_factor
    )

    if all(
        sc.a_atom in reactant.atom_indexes(mol_index=0) for sc in subst_centres
    ):
        attacking_mol = 0
    else:
        attacking_mol = 1

    # Disable the logger to prevent rotation/translations printing
    logger.disabled = True

    # Find the global minimum for inplace rotation, translation and rotation
    min_cost, opt_x = None, None

    for _ in range(n_iters):
        res = minimize(
            get_cost_rotate_translate,
            x0=np.random.random(11),
            method="BFGS",
            tol=0.1,
            args=(reactant, subst_centres, attacking_mol),
        )

        if min_cost is None or res.fun < min_cost:
            min_cost = res.fun
            opt_x = res.x

    # Re-enable the logger
    logger.disabled = False
    logger.info(f"Minimum cost for translating/rotating is {min_cost:.3f}")

    # Translate/rotation the attacking molecule optimally
    reactant.rotate_mol(
        axis=opt_x[:3], theta=opt_x[3], mol_index=attacking_mol
    )
    reactant.translate_mol(vec=opt_x[4:7], mol_index=attacking_mol)
    reactant.rotate_mol(
        axis=opt_x[7:10], theta=opt_x[10], mol_index=attacking_mol
    )

    logger.info("                                                 ... done")

    reactant.atoms.remove_dummy()
    reactant.print_xyz_file()

    return None


@work_in("truncated")
def get_truncated_ts(name, reactant, product, bond_rearr):
    """Get the TS of a truncated reactant and product complex"""

    trnc_reactant = get_truncated_species(reactant, bond_rearr)
    trnc_product = get_truncated_species(product, bond_rearr)

    # Re-find the bond rearrangements, which should exist
    bond_rearrangs = get_bond_rearrangs(trnc_reactant, trnc_product, name=name)

    if bond_rearrangs is None:
        logger.error("Truncation generated a complex with 0 rearrangements")
        return None

    # Find all the possible TSs
    for bond_rearr in bond_rearrangs:
        get_ts(
            name, trnc_reactant, trnc_product, bond_rearr, is_truncated=True
        )

    logger.info("Done with truncation")
    return


def get_ts(name, reactant, product, bond_rearr, is_truncated=False):
    """For a bond rearrangement run PES exploration and TS optimisation to
    find a TS

    ---------------------------------------------------------------------------
    Arguments:
        name (str): Unique identifier for this reaction, used for filenames

        reactant (autode.species.ReactantComplex):

        product (autode.species.ProductComplex):

        bond_rearr (autode.bond_rearrangement.BondRearrangement):

        is_truncated (bool, optional): If the reactant is already truncated
                                       then truncation shouldn't be attempted
                                       and there should be no need to shift
    Returns:
        (autode.transition_states.transition_state.TransitionState): TS
    """

    if bond_rearr.n_fbonds > bond_rearr.n_bbonds:
        raise NotImplementedError(
            "Cannot treat more forming than breaking "
            "bonds, reverse the reaction(?)"
        )

    # If the reaction is a substitution or elimination then the reactants must
    # be orientated correctly, no need to re-rotate/translate if truncated
    if not is_truncated:
        translate_rotate_reactant(
            reactant,
            bond_rearrangement=bond_rearr,
            shift_factor=1.5 if reactant.charge == 0 else 2.5,
        )

    # Reorder the atoms in the product complex so they are equivalent to the
    # reactant
    try:
        mapping = get_mapping(
            graph1=product.graph,
            graph2=reac_graph_to_prod_graph(reactant.graph, bond_rearr),
        )
        product.reorder_atoms(mapping=mapping)
    except NoMapping:
        logger.warning("Could not find the expected bijection R -> P")
        return None

    # If specified then strip non-core atoms from the structure
    if not is_truncated and is_worth_truncating(reactant, bond_rearr):
        get_truncated_ts(name, reactant, product, bond_rearr)

    # There are multiple methods of finding a transition state. Iterate through
    # from the cheapest -> most expensive
    for func, params in ts_guess_funcs_prms(
        name, reactant, product, bond_rearr
    ):
        logger.info(f"Trying to find a TS guess with {func.__name__}")
        ts_guess = func(*params)

        if ts_guess is None:
            continue

        if not ts_guess.could_have_correct_imag_mode:
            continue

        # Form a transition state object and run an OptTS calculation
        ts = TransitionState(ts_guess, bond_rearr=bond_rearr)
        ts.optimise()

        if not ts.is_true_ts:
            continue

        # Save a transition state template if specified in the config
        if Config.make_ts_template:
            ts.save_ts_template(folder_path=Config.ts_template_folder_path)

        logger.info(f"Found a transition state with {func.__name__}")
        return ts

    return None


def _get_ts_neb_from_adaptive_path(
    reactant, product, method, bond_rearr, name, ad_name
):
    from autode.neb import NEB
    from autode.transition_states.ts_guess import TSguess

    if not os.path.exists(f"{ad_name}_path.xyz"):
        logger.warning("Found no adaptive path to generate the NEB from")
        return None

    neb = NEB.from_file(f"{ad_name}_path.xyz")

    if not neb.images.contains_peak:
        logger.info("Adaptive path had no peak – not running a NEB")
        return None

    neb.partition(
        max_delta=Distance(0.2, units="Å"),
        distance_idxs=bond_rearr.active_atoms,
    )
    neb.calculate(
        method=method,
        n_cores=Config.n_cores,
        name_prefix=f"{name}_",
        etol_per_image=PotentialEnergy(0.1, units="kcal mol^-1"),
    )

    if neb.images.contains_peak:
        ts_guess = TSguess(
            atoms=neb.peak_species.atoms,
            reactant=reactant,
            product=product,
            bond_rearr=bond_rearr,
            name=name,
        )
        return ts_guess

    return None
