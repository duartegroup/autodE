from .log import logger
from .config import Config
from .geom import get_neighbour_list
from .geom import get_identical_pairs
from .bond_rearrangement import BondRearrangement
from .bond_rearrangement import gen_equiv_bond_rearrangs
from .substitution import get_complex_xyzs_translated_rotated
from .molecule import Molecule
from .reactions import Dissociation, Rearrangement, Substitution
from .mol_graphs import is_isomorphic
from .mol_graphs import get_adjacency_digraph
from .optts import get_ts
from .template_ts_guess import get_template_ts_guess
from .pes_1d import get_xtb_ts_guess_1dpes_scan
from .pes_1d import get_orca_ts_guess_1dpes_scan


def find_tss(reaction):
    logger.info('Finding possible transition states')
    # TODO elimination reactions
    tss = []

    reactant, product = get_reactant_and_product_complexes(reaction)
    bond_rearrangs = get_bond_rearrangs(reactant, product)
    logger.info('Found *{}* bond rearrangement(s) that lead to products'.format(len(bond_rearrangs)))

    for bond_rearrangement in bond_rearrangs:

        if reaction.type == Substitution:
            reactant.xyzs = get_complex_xyzs_translated_rotated(reactant, reaction.reacs[0], bond_rearrangement)

        ts_guess = get_ts_guess(reaction, reactant, bond_rearrangement)
        ts = get_ts(ts_guess)
        if ts is not None:
            tss.append(ts)

    if len(tss) > 0:
        logger.info('Found *{}* transition state that lead to products'.format(len(tss)))
        return tss

    else:
        logger.error('Did not find any transition state(s)')
        return None


def get_ts_guess(reaction, reactant, bond_rearrang):

    ts_guess_funcs_params = {get_template_ts_guess: (reactant, bond_rearrang.all, reaction.type)}

    if bond_rearrang.n_bbonds == 1 and bond_rearrang.n_fbonds == 0:
        ts_guess_funcs_params[get_xtb_ts_guess_1dpes_scan] = (reactant, bond_rearrang.bbonds[0], 20, reaction.type)
        ts_guess_funcs_params[get_orca_ts_guess_1dpes_scan] = (reactant, bond_rearrang.bbonds[0], 10,
                                                               Config.scan_keywords, 'default', reaction.type)





    # TODO add more functions depending on the number of breaking and forming bonds







    for func, params in ts_guess_funcs_params.items():
        ts_guess = func(*params)
        if ts_guess is not None:
            return ts_guess

    return None


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
        product = gen_two_mol_complex(name='product_complex', mol1=reaction.prods[0], mol2=reaction.prods[1])

    elif reaction.type == Rearrangement:
        reactant = reaction.reacs[0]
        product = reaction.prods[0]

    elif reaction.type == Substitution:
        reactant = gen_two_mol_complex(name='reac_complex', mol1=reaction.reacs[0], mol2=reaction.reacs[1])
        product = gen_two_mol_complex(name='prod_complex', mol1=reaction.prods[0], mol2=reaction.prods[1])

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

    possible_bond_rearrangs = []

    possible_fbonds = mol.get_possible_forming_bonds()
    possible_bbonds = mol.get_possible_breaking_bonds()

    for func in [get_fbonds_bbonds_1b, get_fbonds_bbonds_2b, get_fbonds_bbonds_1b1f, get_fbonds_bbonds_2b1f]:
        possible_bond_rearrangs = func(possible_fbonds, possible_bbonds, mol, product, possible_bond_rearrangs)
        if len(possible_bond_rearrangs) > 0:
            logger.info('Found a molecular graph rearrangement to products with {}'.format(func.__name__))

            if len(possible_bond_rearrangs) > 1:
                logger.info('Multiple possible bond breaking/makings are possible')
                possible_bond_rearrangs = strip_equivalent_bond_rearrangs(mol, possible_bond_rearrangs)

            return possible_bond_rearrangs

    logger.error('Could not find a set of forming/breaking bonds')
    return None


def get_fbonds_bbonds_1b(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):

    for bbond in possible_bbonds:
        rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[], bbonds=[bbond])
        if is_isomorphic(rearranged_graph, product.graph):
            possible_bond_rearrangs.append(BondRearrangement(forming_bonds=[],
                                                             breaking_bonds=[bbond]))

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):

    for i in range(len(possible_bbonds)):
        for j in range(len(possible_bbonds)):
            if i > j:
                bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[], bbonds=[bbond1, bbond2])
                if is_isomorphic(rearranged_graph, product.graph):
                    possible_bond_rearrangs.append(BondRearrangement(forming_bonds=[],
                                                                     breaking_bonds=[bbond1, bbond2]))

    return possible_bond_rearrangs


def get_fbonds_bbonds_1b1f(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):

    for fbond in possible_fbonds:
        for bbond in possible_bbonds:
            rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[fbond], bbonds=[bbond])
            if is_isomorphic(rearranged_graph, product.graph):
                possible_bond_rearrangs.append(BondRearrangement(forming_bonds=[fbond],
                                                                 breaking_bonds=[bbond]))

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b1f(possible_fbonds, possible_bbonds, reactant, product, possible_bond_rearrangs):

    for fbond in possible_fbonds:
        for i in range(len(possible_bbonds)):
            for j in range(len(possible_bbonds)):
                if i > j:
                    bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                    rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[fbond],
                                                                 bbonds=[bbond1, bbond2])

                    if is_isomorphic(rearranged_graph, product.graph):
                        possible_bond_rearrangs.append(BondRearrangement(forming_bonds=[fbond],
                                                                         breaking_bonds=[bbond1, bbond2]))

    return possible_bond_rearrangs


def gen_two_mol_complex(name, mol1, mol2, mol2_shift_ang=100):
    return Molecule(name=name, xyzs=mol1.xyzs + [xyz[:3] + [xyz[3] + mol2_shift_ang] for xyz in mol2.xyzs],
                    solvent=mol1.solvent, charge=(mol1.charge + mol2.charge), mult=(mol1.mult + mol2.mult - 1))


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


def get_nonunique_atoms_and_matches_by_connectivity(mol):
    """
    Generate a dictionary of atom indices in the molecule and their equivalent atoms, as defined by thier connectivity
    e.g. in this function the 3 Hs of CH3 will be equivalent
    :param mol:
    :return:
    """
    nonunique_atoms_and_matches = {}
    adjacency_graphs = [get_adjacency_digraph(atom_i=i, graph=mol.graph) for i in range(mol.n_atoms)]

    for atom_i in range(mol.n_atoms):
        atom_i_matches = []
        for atom_j in range(mol.n_atoms):
            if atom_i != atom_j:                                                        # could be optimised i > j
                if is_isomorphic(adjacency_graphs[atom_i], adjacency_graphs[atom_j]):
                    atom_i_matches.append(atom_j)

        nonunique_atoms_and_matches[atom_i] = atom_i_matches

    return nonunique_atoms_and_matches


def get_nonunique_atoms_and_matches(mol):
    """
    For a molecule and an already generated dictionary of possible non-unique atoms (value = matching atom)
    strip those that could be unique as defined by their neighbor list i.e. a CH3 might have a neighbour list
    for one of the Hs = [c, H, H]  (from closest -> furthest)

    :param mol: (object) Molecule object
    :return: (dict) stripped of what could be non-equivalent atoms
    """

    nonunique_atoms_and_matches = get_nonunique_atoms_and_matches_by_connectivity(mol)

    neighbor_lists = [get_neighbour_list(atom_i=i, mol=mol) for i in range(mol.n_atoms)]

    for atom_i, equiv_atoms in nonunique_atoms_and_matches.items():
        unique_atoms = []
        for atom_k in equiv_atoms:
            if neighbor_lists[atom_i] != neighbor_lists[atom_k]:
                unique_atoms.append(atom_k)

        [equiv_atoms.remove(atom_m) for atom_m in unique_atoms]

    return nonunique_atoms_and_matches


def strip_equivalent_bond_rearrangs(mol, possible_bond_rearrangs):
    """
    Remove any bond rearrangement from possible_bond_rearrangs for which there is already an equivalent in the
    unique_bond_rearrangs list

    :param mol: (object) Molecule object
    :param possible_bond_rearrangs: (list(object)) list of BondRearrangement objects
    :return: (list(object)) list of BondRearrangement objects
    """
    logger.info('Stripping the forming and breaking bond list by discarding symmetry equivs')

    atoms_and_matches = get_nonunique_atoms_and_matches(mol=mol)
    identical_pairs = get_identical_pairs(atoms_and_matches, n_atoms=mol.n_atoms)

    unique_bond_rearrangs = possible_bond_rearrangs[:1]

    for bond_rearrang in possible_bond_rearrangs:
        bond_rearrang_unique = True

        for unique_bond_rearrang in unique_bond_rearrangs:
            all_equiv_bond_rearrgs = gen_equiv_bond_rearrangs(identical_pairs, init_bond_rearrang=unique_bond_rearrang)
            if any([equiv_bond_rearrg == bond_rearrang for equiv_bond_rearrg in all_equiv_bond_rearrgs]):
                bond_rearrang_unique = False
                break

        if bond_rearrang_unique:
            unique_bond_rearrangs.append(bond_rearrang)

    return unique_bond_rearrangs
