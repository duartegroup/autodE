from autode.log import logger
from autode.bond_rearrangement import BondRearrangement
from autode.bond_rearrangement import get_bond_rearrangs
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
import os


def find_tss(reaction, solvent_mol):
    """Finds the transition states of a reaction

    Arguments:
        reaction (reaction object): reaction being examined

    Returns:
        list: list of transition state objects
    """
    logger.info('Finding possible transition states')
    tss = []

    reactant, product = get_reactant_and_product_complexes(reaction)

    logger.info('Looking for a file with bond rearrangements in it')
    if os.path.exists('bond_rearrangs.txt'):
        bond_rearrangs = get_bond_rearrangs_from_file()
    else:
        bond_rearrangs = get_bond_rearrangs(reactant, product)

        if bond_rearrangs is None:
            logger.error('Could not find a set of forming/breaking bonds')
            return None

    save_bond_rearrangs_to_file(bond_rearrangs)
    logger.info(f'Found *{len(bond_rearrangs)}* bond rearrangement(s) that lead to products')

    for bond_rearrangement in bond_rearrangs:
        rearrang_tss = get_ts_obj(reaction, reactant, product, bond_rearrangement, solvent_mol)
        tss += rearrang_tss

    if len(tss) > 0:
        logger.info(f'Found *{len(tss)}* transition state(s) that lead to products')
        return tss

    else:
        logger.error('Did not find any transition state(s)')
        return None


def save_bond_rearrangs_to_file(bond_rearrangs, filename='bond_rearrangs.txt'):
    logger.info('Saving bond rearrangements to bond_rearrangs.txt')
    with open('bond_rearrangs.txt', 'w') as file:
        for bond_rearrang in bond_rearrangs:
            print('fbonds', file=file)
            for fbond in bond_rearrang.fbonds:
                print(*fbond, file=file)
            print('bbonds', file=file)
            for bbond in bond_rearrang.bbonds:
                print(*bbond, file=file)
            print('end', file=file)

    return None


def get_bond_rearrangs_from_file(filename='bond_rearrangs.txt'):
    logger.info('Getting bond rearrangements from file')
    bond_rearrangs = []

    with open(filename, 'r') as br_file:
        fbonds_block = False
        bbonds_block = True
        fbonds = []
        bbonds = []
        for line in br_file:
            if 'fbonds' in line:
                fbonds_block = True
                bbonds_block = False
            if 'bbonds' in line:
                fbonds_block = False
                bbonds_block = True
            if fbonds_block and len(line.split()) == 2:
                atom_id_string = line.split()
                fbonds.append((int(atom_id_string[0]), int(atom_id_string[1])))
            if bbonds_block and len(line.split()) == 2:
                atom_id_string = line.split()
                bbonds.append((int(atom_id_string[0]), int(atom_id_string[1])))
            if 'end' in line:
                bond_rearrangs.append(BondRearrangement(forming_bonds=fbonds, breaking_bonds=bbonds))
                fbonds = []
                bbonds = []

    return bond_rearrangs


def get_ts_guess_funcs_and_params(funcs_params, reaction, reactant, product, bond_rearrang, solvent_mol):

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
                                                       lmethod.scan_keywords, solvent_mol, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))
                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 8, scan_name + '_hl2d', reaction.type, hmethod,
                                                       hmethod.scan_keywords, solvent_mol, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 0:
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d', reaction.type,
                                               hmethod, hmethod.scan_keywords, solvent_mol, delta_bbond_dist)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level',
                                               reaction.type, hmethod, hmethod.opt_keywords, solvent_mol, delta_bbond_dist)))

    if bond_rearrang.n_bbonds >= 1 and bond_rearrang.n_fbonds >= 1:
        for fbond in bond_rearrang.fbonds:
            for bbond in bond_rearrang.bbonds:
                scanned_bonds = [fbond, bbond]
                active_bonds_not_scanned = [bond for bond in bond_rearrang.all if not bond in scanned_bonds]
                scan_name = name + f'_{fbond[0]}-{fbond[1]}_{bbond[0]}-{bbond[1]}'
                delta_fbond_dist = get_avg_bond_length(mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)
                delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)

                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 16, scan_name + '_ll2d', reaction.type, lmethod,
                                                       lmethod.scan_keywords, solvent_mol, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))
                funcs_params.append((get_ts_guess_2d, (reactant, product, fbond, bbond, 8, scan_name + '_hl2d', reaction.type, hmethod,
                                                       hmethod.scan_keywords, solvent_mol, delta_fbond_dist, delta_bbond_dist, active_bonds_not_scanned)))

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 1 and reaction.type in (Substitution, Elimination):
        bbond = bond_rearrang.bbonds[0]
        scan_name = name + f'_{bbond[0]}-{bbond[1]}'
        delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 20, scan_name + '_ll1d_bbond', reaction.type, lmethod,
                                               lmethod.scan_keywords, solvent_mol, delta_bbond_dist, [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_bbond', reaction.type, hmethod,
                                               hmethod.scan_keywords, solvent_mol, delta_bbond_dist, [bond_rearrang.fbonds[0]])))
        funcs_params.append((get_ts_guess_1d, (reactant, product, bbond, 10, scan_name + '_hl1d_opt_level_bbond', reaction.type, hmethod,
                                               hmethod.opt_keywords, solvent_mol, delta_bbond_dist, [bond_rearrang.fbonds[0]])))

    if bond_rearrang.n_bbonds > 0 and bond_rearrang.n_fbonds == 1:
        fbond = bond_rearrang.fbonds[0]
        scan_name = name + f'_{fbond[0]}-{fbond[1]}'
        delta_fbond_dist = get_avg_bond_length(mol=reactant, bond=fbond) - reactant.calc_bond_distance(fbond)

        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 20, scan_name + '_ll1d_fbond', reaction.type, lmethod,
                                               lmethod.scan_keywords,  solvent_mol, delta_fbond_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_fbond', reaction.type, hmethod,
                                               hmethod.scan_keywords, solvent_mol, delta_fbond_dist, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_1d, (reactant, product, fbond, 10, scan_name + '_hl1d_opt_level_fbond', reaction.type, hmethod,
                                               hmethod.opt_keywords, solvent_mol, delta_fbond_dist, bond_rearrang.bbonds)))

    if bond_rearrang.n_fbonds == 2:
        fbond1, fbond2 = bond_rearrang.fbonds
        scan_name = name + f'_{fbond1[0]}-{fbond1[1]}_{fbond2[0]}-{fbond2[1]}'
        delta_fbond_dist1 = get_avg_bond_length(mol=reactant, bond=fbond1) - reactant.calc_bond_distance(fbond1)
        delta_fbond_dist2 = get_avg_bond_length(mol=reactant, bond=fbond2) - reactant.calc_bond_distance(fbond2)
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 16, scan_name + '_ll2d_fbonds', reaction.type, lmethod,
                                               lmethod.scan_keywords, solvent_mol, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, fbond1, fbond2, 8, scan_name + '_hl2d_fbonds', reaction.type, hmethod,
                                               hmethod.scan_keywords, solvent_mol, delta_fbond_dist1, delta_fbond_dist2, bond_rearrang.bbonds)))

    if bond_rearrang.n_bbonds == 2:
        bbond1, bbond2 = bond_rearrang.bbonds
        delta_bbond_dist = get_bbond_dist(reaction, solvent_mol)
        scan_name = name + f'_{bbond1[0]}-{bbond1[1]}_{bbond2[0]}-{bbond2[1]}'
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 16, scan_name + '_ll2d_bbonds', reaction.type, lmethod,
                                               lmethod.scan_keywords, solvent_mol, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))
        funcs_params.append((get_ts_guess_2d, (reactant, product, bbond1, bbond2, 8, scan_name + '_hl2d_bbonds', reaction.type, hmethod,
                                               hmethod.scan_keywords, solvent_mol, delta_bbond_dist, delta_bbond_dist, bond_rearrang.fbonds)))

    return funcs_params


def get_reactant_and_product_complexes(reaction):
    """Find the reactant and product complexes for a reaction, i.e. for a substitution then these will have both reactants
    and products in the same xyzs, whereas for a Dissociation the reactant will be the sole reactant

    Arguments:
        reaction (reaction object): reaction object

    Returns:
        tuple: (reactant molecule object, product molecule object)
    """

    reactant, product = None, None

    if reaction.type == Dissociation:
        reactant = reaction.reacs[0]
        product = gen_two_mol_complex(name='product_complex', mol1=reaction.prods[0], mol2=reaction.prods[1])

    elif reaction.type == Rearrangement:
        reactant = reaction.reacs[0]
        product = reaction.prods[0]

    elif reaction.type == Substitution:
        reactant = gen_two_mol_complex(name='reac_complex', mol1=reaction.reacs[0], mol2=reaction.reacs[1])
        product = gen_two_mol_complex(name='prod_complex', mol1=reaction.prods[0], mol2=reaction.prods[1])

    elif reaction.type == Elimination:
        reactant = gen_two_mol_complex(name='reac_complex', mol1=reaction.reacs[0], mol2=reaction.reacs[1])
        product = gen_three_mol_complex(name='prod_complex', mol1=reaction.prods[0], mol2=reaction.prods[1], mol3=reaction.prods[2])

    else:
        logger.critical('Reaction type not currently supported')
        exit()

    return reactant, product


def gen_two_mol_complex(name, mol1, mol2, mol2_shift_ang=100):
    complex_mol = Molecule(name=name, xyzs=mol1.xyzs + [xyz[:3] + [xyz[3] + mol2_shift_ang] for xyz in mol2.xyzs],
                           solvent=mol1.solvent, charge=(mol1.charge + mol2.charge), mult=(mol1.mult + mol2.mult - 1))
    complex_mol.charges = mol1.charges + mol2.charges
    complex_mol.stereocentres = get_stereoatoms([mol1, mol2])
    complex_mol.pi_bonds = get_pi_bonds([mol1, mol2])
    return complex_mol


def gen_three_mol_complex(name, mol1, mol2, mol3, mol2_shift_ang=100, mol3_shift_ang=-100):
    complex_mol = Molecule(name=name, xyzs=mol1.xyzs + [xyz[:3] + [xyz[3] + mol2_shift_ang] for xyz in mol2.xyzs] + [xyz[:3] + [xyz[3] + mol3_shift_ang] for xyz in mol3.xyzs],
                           solvent=mol1.solvent, charge=(mol1.charge + mol2.charge + mol3.charge), mult=(mol1.mult + mol2.mult + mol3.mult - 2))
    complex_mol.charges = mol1.charges + mol2.charges + mol3.charges
    complex_mol.stereocentres = get_stereoatoms([mol1, mol2, mol3])
    complex_mol.pi_bonds = get_pi_bonds([mol1, mol2, mol3])
    return complex_mol


def get_stereoatoms(mols):
    n_atoms = 0
    stereocentres = []
    for mol in mols:
        if mol.stereocentres is not None:
            for stereocentre in mol.stereocentres:
                stereocentres.append(stereocentre + n_atoms)
        n_atoms += mol.n_atoms
    if len(stereocentres) > 0:
        return stereocentres
    else:
        return None


def get_pi_bonds(mols):
    n_atoms = 0
    pi_bonds = []
    for mol in mols:
        if mol.pi_bonds is not None:
            for pi_bond in mol.pi_bonds:
                pi_bonds.append((pi_bond[0] + n_atoms, pi_bond[1] + n_atoms))
        n_atoms += mol.n_atoms
    if len(pi_bonds) > 0:
        return pi_bonds
    else:
        return None


def get_ts_obj(reaction, reactant, product, bond_rearrangement, solvent_mol, strip_molecule=True):
    if solvent_mol is not None:
        strip_molecule = False
    tss = []

    active_atoms = set()
    for active_atom in bond_rearrangement.active_atoms:
        active_atoms.add(active_atom)

    reactant.active_atoms = sorted(active_atoms)

    # get the product graph with the atom indices of the reactant
    full_prod_graph_reac_indices = mol_graphs.reac_graph_to_prods(reactant.graph, bond_rearrangement)
    # mapping[product indices] = reactant indices
    full_mapping = mol_graphs.get_mapping(product.graph, full_prod_graph_reac_indices)
    inv_full_mapping = {v: k for k, v in full_mapping.items()}

    print(inv_full_mapping)

    exit()

    # ensure any formed stereocentres are included in ts stereocentres
    formed_stereocentres = []
    if product.stereocentres is not None:
        if reactant.stereocentres is None:
            reactant.stereocentres = []
        for stereocentre in product.stereocentres:
            reac_index = full_mapping[stereocentre]
            if not reac_index in reactant.stereocentres:
                formed_stereocentres.append(reac_index)
        if len(formed_stereocentres) > 0:
            reactant.stereocentres += formed_stereocentres

    if reaction.type in [Substitution, Elimination]:
        if any(mol.charge != 0 for mol in reaction.reacs) and solvent_mol is None:
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
        tss = get_ts_obj(reaction, reactant, product, bond_rearrangement, solvent_mol, strip_molecule=False)

    return tss


def get_bbond_dist(reaction, solvent_mol):
    if any(mol.charge != 0 for mol in reaction.prods) and solvent_mol is None:
        return 2.5
    else:
        return 1.5
