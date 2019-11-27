from autode.log import logger
from autode.atoms import get_maximal_valance
from autode.bond_rearrangement import BondRearrangement
from autode.substitution import set_complex_xyzs_translated_rotated
from autode.molecule import Molecule
from autode.reactions import Dissociation, Rearrangement, Substitution, Elimination
from autode.bond_lengths import get_avg_bond_length
from autode import mol_graphs
from autode.transition_states.optts import get_ts
from autode.transition_states.template_ts_guess import get_template_ts_guess
from autode.pes_1d import get_ts_guess_1d
from autode.pes_2d import get_ts_guess_2d
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.transition_states.transition_state import TS
import numpy as np
import os


def find_tss(reaction):
    logger.info('Finding possible transition states')
    tss = []

    reactant, product = get_reactant_and_product_complexes(reaction)

    logger.info('Looking for a file with bond rearrangements in it')
    if os.path.exists('bond_rearrangs.txt'):
        logger.info('Getting bond rearrangements from file')
        bond_rearrangs = []
        with open('bond_rearrangs.txt', 'r') as file:
            fbonds_block = False
            bbonds_block = True
            fbonds = []
            bbonds = []
            for line in file:
                if 'fbonds' in line:
                    fbonds_block = True
                    bbonds_block = False
                if 'bbonds' in line:
                    fbonds_block = False
                    bbonds_block = True
                if fbonds_block and len(line.split()) == 2:
                    atom_id_string = line.split()
                    fbonds.append(
                        (int(atom_id_string[0]), int(atom_id_string[1])))
                if bbonds_block and len(line.split()) == 2:
                    atom_id_string = line.split()
                    bbonds.append(
                        (int(atom_id_string[0]), int(atom_id_string[1])))
                if 'end' in line:
                    bond_rearrangs.append(BondRearrangement(
                        forming_bonds=fbonds, breaking_bonds=bbonds))
                    fbonds = []
                    bbonds = []
    else:
        bond_rearrangs = get_bond_rearrangs(reactant, product)
        logger.info('Saving bond rearrangements to bond_rearrange.txt')
        with open('bond_rearrangs.txt', 'w') as file:
            for bond_rearrang in bond_rearrangs:
                print('fbonds', file=file)
                for fbond in bond_rearrang.fbonds:
                    print(*fbond, file=file)
                print('bbonds', file=file)
                for bbond in bond_rearrang.bbonds:
                    print(*bbond, file=file)
                print('end', file=file)

    if bond_rearrangs is None:
        logger.error('Could not find a set of forming/breaking bonds')
        return None

    reactant_pi_systems = []
    reac_no_atoms = [reac.n_atoms for reac in reaction.reacs]
    for reac_no, reac in enumerate(reaction.reacs):
        if reac.pi_systems is not None:
            # indices will change in the reac complex
            no_atoms = sum(reac_no_atoms[:reac_no])
            for system in reac.pi_systems:
                new_system = [atom + no_atoms for atom in system]
                reactant_pi_systems.append(new_system)

    reactant.pi_systems = reactant_pi_systems

    logger.info(
        f'Found *{len(bond_rearrangs)}* bond rearrangement(s) that lead to products')

    for bond_rearrangement in bond_rearrangs:
        rearrang_tss = get_ts_obj(
            reaction, reactant, product, bond_rearrangement)
        tss += rearrang_tss

    if len(tss) > 0:
        logger.info(
            f'Found *{len(tss)}* transition state(s) that lead to products')
        return tss

    else:
        logger.error('Did not find any transition state(s)')
        return None


def get_ts_guess_funcs_and_params(funcs_params, reaction, reactant, product, bond_rearrang):

    name = '+'.join([r.name for r in reaction.reacs]) + '--' + \
        '+'.join([p.name for p in reaction.prods])

    if reactant.is_fragment:
        name += 'fragment_'

    lmethod, hmethod = get_lmethod(), get_hmethod()

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 0:
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        delta_bbond_dist = get_bbond_dist(reaction)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 20, scan_name + '_ll1d',
                                               reaction.type, lmethod, lmethod.scan_keywords, delta_bbond_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d',
                                               reaction.type, hmethod, hmethod.scan_keywords, delta_bbond_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level',
                                               reaction.type, hmethod, hmethod.opt_keywords, delta_bbond_dist)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 1 and reaction.type in (Substitution, Elimination):
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        delta_bbond_dist = get_bbond_dist(reaction)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 20, scan_name + '_ll1d_bbond',
                                               reaction.type, lmethod, lmethod.scan_keywords, delta_bbond_dist,
                                               [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_bbond',
                                               reaction.type, hmethod, hmethod.scan_keywords, delta_bbond_dist,
                                               [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level_bbond',
                                               reaction.type, hmethod, hmethod.opt_keywords, delta_bbond_dist,
                                               [bond_rearrang.fbonds[0]])))

    if bond_rearrang.n_bbonds > 0 and bond_rearrang.n_fbonds == 1:
        fbond = bond_rearrang.fbonds[0]
        scan_name = name + f'_{fbond[0]}-{fbond[1]}'
        delta_fbond_dist = get_avg_bond_length(
            mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)

        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 20, scan_name + '_ll1d_fbond', reaction.type, lmethod,
                                               lmethod.scan_keywords, delta_fbond_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_fbond', reaction.type, hmethod,
                                               hmethod.scan_keywords, delta_fbond_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_opt_level_fbond', reaction.type, hmethod,
                                               hmethod.opt_keywords, delta_fbond_dist, bond_rearrang.bbonds)))

    if bond_rearrang.n_bbonds >= 1 and bond_rearrang.n_fbonds >= 1:
        for fbond in bond_rearrang.fbonds:
            for bbond in bond_rearrang.bbonds:
                scanned_bonds = [fbond, bbond]
                active_bonds_not_scanned = [
                    bond for bond in bond_rearrang.all if not bond in scanned_bonds]
                scan_name = name + \
                    f'_{fbond[0]}-{fbond[1]}_{bbond[0]}-{bbond[1]}'
                delta_fbond_dist = get_avg_bond_length(
                    mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)
                delta_bbond_dist = get_bbond_dist(reaction)

                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 16, scan_name + '_ll2d', reaction.type, lmethod,
                                                       lmethod.scan_keywords, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))
                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 8, scan_name + '_hl2d', reaction.type, hmethod,
                                                       hmethod.scan_keywords, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))

    if bond_rearrang.n_fbonds == 2:
        fbond1, fbond2 = bond_rearrang.fbonds
        scan_name = name + f'_{fbond1[0]}-{fbond1[1]}_{fbond2[0]}-{fbond2[1]}'
        delta_fbond_dist1 = get_avg_bond_length(
            mol=reactant, bond=fbond1) - reactant.calc_bond_distance(fbond1)
        delta_fbond_dist2 = get_avg_bond_length(
            mol=reactant, bond=fbond2) - reactant.calc_bond_distance(fbond2)
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 16, scan_name + '_ll2d_fbonds', reaction.type, lmethod,
                                               lmethod.scan_keywords, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 8, scan_name + '_hl2d_fbonds', reaction.type, hmethod,
                                               hmethod.scan_keywords, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))

    if bond_rearrang.n_bbonds == 2:
        bbond1, bbond2 = bond_rearrang.bbonds
        delta_bbond_dist = get_bbond_dist(reaction)
        scan_name = name + f'_{bbond1[0]}-{bbond1[1]}_{bbond2[0]}-{bbond2[1]}'
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 16, scan_name + '_ll2d_bbonds', reaction.type, lmethod,
                                               lmethod.scan_keywords, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 8, scan_name + '_hl2d_bbonds', reaction.type, hmethod,
                                               hmethod.scan_keywords, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))

    # if bond_rearrang.n_fbonds == 2 and bond_rearrang.n_bbonds == 1:
    #     fbond1, fbond2 = bond_rearrang.fbonds
    #     bbond = bond_rearrang.bbonds[0]
    #     scan_name = name + \
    #         f'_{fbond1[0]}-{fbond1[1]}_{fbond2[0]}-{fbond2[1]}_{bbond[0]}-{bbond[1]}'
    #     delta_fbond_dist1 = get_avg_bond_length(
    #         mol=reactant, bond=fbond1) - reactant.calc_bond_distance(fbond1)
    #     delta_fbond_dist2 = get_avg_bond_length(
    #         mol=reactant, bond=fbond2) - reactant.calc_bond_distance(fbond2)
    #     delta_bbond_dist = get_bbond_dist(reaction)
    #     funcs_params.append((get_ts_guess_3d, (reactant, product, fbond1, fbond2, bbond, 8, scan_name + '_ll3d_bbonds', reaction.type, lmethod,
    #                                            lmethod.scan_keywords, delta_fbond_dist1, delta_fbond_dist2, delta_bbond_dist, bond_rearrang.fbonds)))

    # if bond_rearrang.n_fbonds == 1 and bond_rearrang.n_bbonds == 2:
    #     fbond = bond_rearrang.fbonds[0]
    #     bbond1, bbond2 = bond_rearrang.bbonds
    #     scan_name = name + \
    #         f'_{fbond[0]}-{fbond[1]}_{bbond1[0]}-{bbond1[1]}_{bbond2[0]}-{bbond2[1]}'
    #     delta_fbond_dist = get_avg_bond_length(
    #         mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)
    #     delta_bbond_dist = get_bbond_dist(reaction)
    #     funcs_params.append((get_ts_guess_3d, (reactant, product, fbond, bbond1, bbond2, 8, scan_name + '_ll3d_bbonds', reaction.type, lmethod,
    #                                            lmethod.scan_keywords, delta_fbond_dist, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))

    return funcs_params


def get_reactant_and_product_complexes(reaction):
    """
    Find the reactant and product complexes for a reaction, i.e. for a substitution then these will have both reactants
    and products in the same xyzs, whereas for a Dissociation the reactant will be the sole reactant
    :param reaction:
    :return:
    """

    reactant, product = None, None

    if reaction.type == Dissociation:
        reactant = reaction.reacs[0]
        product = gen_two_mol_complex(
            name='product_complex', mol1=reaction.prods[0], mol2=reaction.prods[1])

    elif reaction.type == Rearrangement:
        reactant = reaction.reacs[0]
        product = reaction.prods[0]

    elif reaction.type == Substitution:
        reactant = gen_two_mol_complex(
            name='reac_complex', mol1=reaction.reacs[0], mol2=reaction.reacs[1])
        product = gen_two_mol_complex(
            name='prod_complex', mol1=reaction.prods[0], mol2=reaction.prods[1])

    elif reaction.type == Elimination:
        reactant = gen_two_mol_complex(
            name='reac_complex', mol1=reaction.reacs[0], mol2=reaction.reacs[1])
        product = gen_three_mol_complex(
            name='prod_complex', mol1=reaction.prods[0], mol2=reaction.prods[1], mol3=reaction.prods[2])

    else:
        logger.critical('Reaction type not currently supported')
        exit()

    return reactant, product


def get_bond_rearrangs(mol, product):
    """
    For a reactant and product (complex) find the set of breaking and forming bonds that will turn reactants into
    products. Will run O(n), O(n^2), O(n^2), O(n^3) number of molecular graph isomorphisms. This could be slow...
    :param mol: (object) Molecule object
    :param product: (object) Molecule object
    :return:
    """
    logger.info('Finding the possible forming and breaking bonds')

    possible_bond_rearrangements = []

    possible_fbonds = mol.get_possible_forming_bonds()
    possible_bbonds = mol.get_possible_breaking_bonds()

    delta_n_bonds = mol.n_bonds - product.n_bonds
    if delta_n_bonds == 0:
        funcs = [get_fbonds_bbonds_1b1f, get_fbonds_bbonds_2b2f]
    elif delta_n_bonds == 1:
        funcs = [get_fbonds_bbonds_1b, get_fbonds_bbonds_2b1f]
    elif delta_n_bonds == 2:
        funcs = [get_fbonds_bbonds_2b]
    elif delta_n_bonds == -1:
        funcs = [get_fbonds_bbonds_1b2f]
    else:
        logger.error(
            f'Cannot treat a change in bonds reactant <- product of {delta_n_bonds}')
        return None

    for func in funcs:
        possible_bond_rearrangements = func(
            possible_fbonds, possible_bbonds, mol, product, possible_bond_rearrangements)
        if len(possible_bond_rearrangements) > 0:
            logger.info(
                f'Found a molecular graph rearrangement to products with {func.__name__}')
            # This function will return with from the first bond rearrangement that leads to products

            n_bond_rearrangs = len(possible_bond_rearrangements)
            if n_bond_rearrangs > 1:
                logger.info(
                    f'Multiple *{n_bond_rearrangs}* possible bond breaking/makings are possible')
                possible_bond_rearrangements = strip_equivalent_bond_rearrangs(
                    mol, possible_bond_rearrangements)

            return possible_bond_rearrangements

    return None


def add_bond_rearrangment(bond_rearrangs, reactant, product, fbonds, bbonds):

    # Check that the bond rearrangement doesn't exceed standard atom valances
    bbond_atoms = [atom for bbond in bbonds for atom in bbond]
    for fbond in fbonds:
        for atom in fbond:
            atom_label = reactant.get_atom_label(atom)
            if reactant.graph.degree(atom) == get_maximal_valance(atom_label) and atom not in bbond_atoms:
                # If we are here then there is at least one atom that will exceed it's maximal valance, therefore
                # we don't need to run isomorphism
                return bond_rearrangs

    rearranged_graph = generate_rearranged_graph(
        reactant.graph, fbonds=fbonds, bbonds=bbonds)
    if mol_graphs.is_isomorphic(rearranged_graph, product.graph):
        bond_rearrangs.append(BondRearrangement(
            forming_bonds=fbonds, breaking_bonds=bbonds))

    return bond_rearrangs


def get_fbonds_bbonds_1b(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):

    for bbond in possible_bbonds:
        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                        fbonds=[], bbonds=[bbond])
    return possible_bond_rearrangs


def get_fbonds_bbonds_2b(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):
    logger.info(f'Have {len(possible_bbonds)**2} isomorphisms to do')

    for i in range(len(possible_bbonds)):
        for j in range(len(possible_bbonds)):
            if i > j:
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                                fbonds=[],
                                                                bbonds=[possible_bbonds[i], possible_bbonds[j]])
    return possible_bond_rearrangs


def get_fbonds_bbonds_1b1f(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):

    logger.info(
        f'Have {len(possible_bbonds)*len(possible_fbonds)} isomorphisms to do')

    for fbond in possible_fbonds:
        for bbond in possible_bbonds:
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                            fbonds=[fbond], bbonds=[bbond])

    return possible_bond_rearrangs


def get_fbonds_bbonds_1b2f(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):
    logger.info(
        f'Have {len(possible_bbonds)*len(possible_fbonds)**2} isomorphisms to do')

    for bbond in possible_bbonds:
        for i in range(len(possible_fbonds)):
            for j in range(len(possible_fbonds)):
                if i > j:
                    fbond1, fbond2 = possible_fbonds[i], possible_fbonds[j]
                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                                    fbonds=[fbond1, fbond2], bbonds=[bbond])
    return possible_bond_rearrangs


def get_fbonds_bbonds_2b1f(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):
    logger.info(
        f'Have {len(possible_bbonds)**2*len(possible_fbonds)} isomorphisms to do')

    for fbond in possible_fbonds:
        for i in range(len(possible_bbonds)):
            for j in range(len(possible_bbonds)):
                if i > j:
                    bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                                    fbonds=[fbond], bbonds=[bbond1, bbond2])
    return possible_bond_rearrangs


def get_fbonds_bbonds_2b2f(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):
    logger.info('Getting possible 2 breaking and 2 forming bonds')
    logger.info(
        f'Have {len(possible_bbonds)**2*len(possible_fbonds)**2} isomorphisms to do')

    for m in range(len(possible_fbonds)):
        for n in range(len(possible_fbonds)):
            if m > n:
                for i in range(len(possible_bbonds)):
                    for j in range(len(possible_bbonds)):
                        if i > j:
                            bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                            fbond1, fbond2 = possible_fbonds[m], possible_fbonds[n]
                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                                            fbonds=[
                                                                                fbond1, fbond2],
                                                                            bbonds=[bbond1, bbond2])
    return possible_bond_rearrangs


def gen_two_mol_complex(name, mol1, mol2, mol2_shift_ang=100):
    return Molecule(name=name, xyzs=mol1.xyzs + [xyz[:3] + [xyz[3] + mol2_shift_ang] for xyz in mol2.xyzs],
                    solvent=mol1.solvent, charge=(mol1.charge + mol2.charge), mult=(mol1.mult + mol2.mult - 1))


def gen_three_mol_complex(name, mol1, mol2, mol3, mol2_shift_ang=100, mol3_shift_ang=-100):
    return Molecule(name=name, xyzs=mol1.xyzs + [xyz[:3] + [xyz[3] + mol2_shift_ang] for xyz in mol2.xyzs] + [xyz[:3] + [xyz[3] + mol3_shift_ang] for xyz in mol3.xyzs],
                    solvent=mol1.solvent, charge=(mol1.charge + mol2.charge + mol3.charge), mult=(mol1.mult + mol2.mult + mol3.mult - 2))


def generate_rearranged_graph(graph, fbonds, bbonds):
    """
    Generate a rearranged graph by breaking bonds (edge) and forming others (edge)
    :param graph: (nx graph object)
    :param fbonds: (tuple) Forming bond ids
    :param bbonds: (tuple) Breaking bond ids
    :return:
    """
    rearranged_graph = graph.copy()
    for fbond in fbonds:
        rearranged_graph.add_edge(*fbond)
    for bbond in bbonds:
        rearranged_graph.remove_edge(*bbond)

    return rearranged_graph


def strip_equivalent_bond_rearrangs(mol, possible_bond_rearrangs, depth=6):
    """
    Remove any bond rearrangement from possible_bond_rearrangs for which there is already an equivalent in the
    unique_bond_rearrangements list

    :param mol: (object) Molecule object
    :param depth: (int) Depth of neighbour list that must be identical for a set of atoms to be considered equivalent
    :param possible_bond_rearrangs: (list(object)) list of BondRearrangement objects
    :return: (list(object)) list of BondRearrangement objects
    """
    logger.info(
        'Stripping the forming and breaking bond list by discarding rearrangements with equivalent atoms')

    unique_bond_rearrangements = []

    for bond_rearrang in possible_bond_rearrangs:
        bond_rearrang_is_unique = True

        # Compare bond_rearrang to all those already considered to be unique,
        for unique_bond_rearrang in unique_bond_rearrangements:

            if (unique_bond_rearrang.get_active_atom_neighbour_lists(mol=mol, depth=depth) ==
                    bond_rearrang.get_active_atom_neighbour_lists(mol=mol, depth=depth)):
                bond_rearrang_is_unique = False

        if bond_rearrang_is_unique:
            unique_bond_rearrangements.append(bond_rearrang)

    logger.info(
        f'Stripped {len(possible_bond_rearrangs)-len(unique_bond_rearrangements)} bond rearrangements')
    return unique_bond_rearrangements


def get_ts_obj(reaction, reactant, product, bond_rearrangement, strip_molecule=True):
    tss = []
    if reaction.type in [Substitution, Elimination]:
        if any(mol.charge != 0 for mol in reaction.reacs):
            shift_factor = 3
        else:
            shift_factor = 2
        set_complex_xyzs_translated_rotated(
            reactant, reaction.reacs, bond_rearrangement, shift_factor)

    reactant_core_atoms = None

    active_atoms = set()
    for active_atom in bond_rearrangement.active_atoms:
        active_atoms.add(active_atom)

    reactant.active_atoms = sorted(active_atoms)

    if strip_molecule:
        # get the product graph with the atom indices of the reactant
        full_prod_graph_reac_indices = mol_graphs.reac_graph_to_prods(
            reactant.graph, bond_rearrangement)

        reactant_core_atoms = reactant.get_core_atoms(
            full_prod_graph_reac_indices)
        product_core_atoms = mol_graphs.get_product_core_atoms(
            product, full_prod_graph_reac_indices)

    reac_mol, reac_mol_rearrangement = reactant.strip_core(
        reactant_core_atoms, bond_rearrangement)

    if reac_mol.is_fragment:
        stripped_prod_graph = mol_graphs.reac_graph_to_prods(
            reac_mol.graph, reac_mol_rearrangement)
        product_core_atoms = mol_graphs.get_product_core_atoms(
            product, stripped_prod_graph)
    else:
        product_core_atoms = None

    prod_mol, _ = product.strip_core(product_core_atoms)

    funcs_params = [
        (get_template_ts_guess, (reactant, bond_rearrangement.all, reaction.type, product))]

    for func, params in get_ts_guess_funcs_and_params(funcs_params, reaction, reac_mol, prod_mol, reac_mol_rearrangement):
        logger.info(f'Trying to find a TS guess with {func.__name__}')
        ts_guess = func(*params)

        get_ts_output = get_ts(ts_guess)

        if get_ts_output is not None:
            ts = TS(get_ts_output[0], converged=get_ts_output[1])
            if ts.is_true_ts():
                logger.info(
                    f'Found a transition state with {func.__name__}')
                if reac_mol.is_fragment:
                    logger.info('Finding full TS')
                    ts_guess_with_decoratation = get_template_ts_guess(
                        reactant, bond_rearrangement.all, reaction.type, product)
                    full_get_ts_output = get_ts(
                        ts_guess_with_decoratation)
                    if full_get_ts_output is not None:
                        ts_with_decoratation = TS(
                            full_get_ts_output[0], converged=full_get_ts_output[1])
                        if ts_with_decoratation.is_true_ts():
                            logger.info('Found full TS')
                            tss.append(ts_with_decoratation)
                else:
                    tss.append(ts)
                break

    if len(tss) == 0 and reac_mol.is_fragment:
        logger.info(
            'Found no transition states using the fragment, will try with the whole molecule')
        tss = get_ts_obj(reaction, reactant, product,
                         bond_rearrangement, strip_molecule=False)

    return tss


def get_bbond_dist(reaction):
    if any(mol.charge != 0 for mol in reaction.prods):
        return 2.5
    else:
        return 1.5
