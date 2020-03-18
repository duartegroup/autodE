from autode.geom import get_neighbour_list
from autode.log import logger
from autode import mol_graphs
from autode.atoms import get_maximal_valance
from autode.mol_graphs import is_isomorphic
import itertools
import os


def get_bond_rearrangs(reactant, product, name):
    """For a reactant and product (complex) find the set of breaking and forming bonds that will turn reactants into
    products. This works by determining the types of bonds that have been made/broken (i.e CH) and then only considering
    rearrangements involving those bonds. 

    Arguments:
        reactant (autode.complex.ReactantComplex):
        product (autode.complex.ProductComplex):
        name (str): Name of the reaction

    Returns:
        list: list of bond rearrang objects linking reacs and prods
    """
    logger.info('Finding the possible forming and breaking bonds')

    if os.path.exists(f'{name}_bond_rearrangs.txt'):
        return get_bond_rearrangs_from_file(filename=f'{name}_bond_rearrangs.txt')

    if is_isomorphic(reactant.graph, product.graph) and product.n_atoms > 3:
        logger.error('Reactant (complex) is isomorphic to product (complex). Bond rearrangement '
                     'cannot be determined unless the substrates are limited in size')
        return None

    possible_bond_rearrangements = []

    reac_bond_dict = mol_graphs.get_bond_type_list(reactant.graph)
    prod_bond_dict = mol_graphs.get_bond_type_list(product.graph)

    # list of lists, shallow level separates by type of bond, deeper level is bonds of those type that can break
    all_possible_bbonds = []
    # list of lists, shallow level separates by type of bond, deeper level is bonds of those type that can form
    all_possible_fbonds = []
    # list of tuples, shallow level separates by type of bond, tuple is ([bbonds of that type], [fbonds of that type])
    possible_fbond_and_bbonds = []
    # list of bonds that can be broken of the type of forming bond
    fbond_atom_type_bbonds = None
    # list of bonds that can be formed of the type of breaking bond
    bbond_atom_type_fbonds = None

    for reac_key, reac_bonds in reac_bond_dict.items():
        prod_bonds = prod_bond_dict[reac_key]
        possible_fbonds = mol_graphs.get_fbonds(reactant.graph, reac_key)
        if len(prod_bonds) < len(reac_bonds):
            all_possible_bbonds.append(reac_bonds)
            bbond_atom_type_fbonds = possible_fbonds
        elif len(prod_bonds) > len(reac_bonds):
            all_possible_fbonds.append(possible_fbonds)
            fbond_atom_type_bbonds = reac_bonds
        else:
            if len(reac_bonds) != 0:
                possible_fbond_and_bbonds.append([reac_bonds, possible_fbonds])

    delta_n_bonds = reactant.graph.number_of_edges() - product.graph.number_of_edges()
    if delta_n_bonds == 0:
        funcs = [get_fbonds_bbonds_1b1f, get_fbonds_bbonds_2b2f]
    elif delta_n_bonds == 1:
        funcs = [get_fbonds_bbonds_1b, get_fbonds_bbonds_2b1f]
    elif delta_n_bonds == 2:
        funcs = [get_fbonds_bbonds_2b]
    elif delta_n_bonds == -1:
        funcs = [get_fbonds_bbonds_1b2f]
    else:
        logger.error(f'Cannot treat a change in bonds reactant <- product of {delta_n_bonds}')
        return None

    for func in funcs:
        possible_bond_rearrangements = func(reactant, product, possible_bond_rearrangements, all_possible_bbonds, all_possible_fbonds,
                                            possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds)
        if len(possible_bond_rearrangements) > 0:
            logger.info(f'Found a molecular graph rearrangement to products with {func.__name__}')
            # This function will return with from the first bond rearrangement that leads to products

            n_bond_rearrangs = len(possible_bond_rearrangements)
            if n_bond_rearrangs > 1:
                logger.info(f'Multiple *{n_bond_rearrangs}* possible bond breaking/makings are possible')
                possible_bond_rearrangements = strip_equivalent_bond_rearrangs(reactant, possible_bond_rearrangements)

            save_bond_rearrangs_to_file(possible_bond_rearrangements, filename=f'{name}_bond_rearrangs.txt')

            logger.info(f'Found *{len(possible_bond_rearrangements)}* bond rearrangement(s) that lead to products')
            return possible_bond_rearrangements

    return None


def save_bond_rearrangs_to_file(bond_rearrangs, filename='bond_rearrangs.txt'):
    logger.info(f'Saving bond rearrangements to {filename}')
    with open(filename, 'w') as file:
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

    if not os.path.exists(filename):
        logger.error('No bond rearrangments file')
        return None

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


def add_bond_rearrangment(bond_rearrangs, reactant, product, fbonds, bbonds):
    """For a possible bond rearrangement, sees if the products are made, and adds it to the bond rearrang list if it does

    Arguments:
        bond_rearrangs (list): list of working bond rearrangments
        reactant (molecule object): reactant complex
        product (molecule object): product complex
        fbonds (list of tuples): list of bonds to be made
        bbonds (list of tuples): list of bonds to be broken

    Returns:
        list: updated list of working bond rearrangments
    """

    # Check that the bond rearrangement doesn't exceed standard atom valances
    bbond_atoms = [atom for bbond in bbonds for atom in bbond]
    for fbond in fbonds:
        for atom in fbond:
            atom_label = reactant.atoms[atom].label
            if reactant.graph.degree(atom) == get_maximal_valance(atom_label) and atom not in bbond_atoms:
                # If we are here then there is at least one atom that will exceed it's maximal valance, therefore
                # we don't need to run isomorphism
                return bond_rearrangs

    rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=fbonds, bbonds=bbonds)
    if mol_graphs.is_isomorphic(rearranged_graph, product.graph):
        ordered_fbonds = []
        ordered_bbonds = []
        for fbond in fbonds:
            if fbond[0] < fbond[1]:
                ordered_fbonds.append((fbond[0], fbond[1]))
            else:
                ordered_fbonds.append((fbond[1], fbond[0]))
        for bbond in bbonds:
            if bbond[0] < bbond[1]:
                ordered_bbonds.append((bbond[0], bbond[1]))
            else:
                ordered_bbonds.append((bbond[1], bbond[0]))
        ordered_fbonds.sort()
        ordered_bbonds.sort()
        bond_rearrangs.append(BondRearrangement(forming_bonds=ordered_fbonds, breaking_bonds=ordered_bbonds))

    return bond_rearrangs


def generate_rearranged_graph(graph, fbonds, bbonds):
    """Generate a rearranged graph by breaking bonds (edge) and forming others (edge)

    Arguments:
        graph (nx.Graph): reactant graph
        fbonds (list(tuple)): list of bonds to be made
        bbonds (list(tuple)): list of bonds to be broken

    Returns:
        nx.Graph: rearranged graph
    """
    rearranged_graph = graph.copy()
    for fbond in fbonds:
        rearranged_graph.add_edge(*fbond)
    for bbond in bbonds:
        rearranged_graph.remove_edge(*bbond)

    return rearranged_graph


def get_fbonds_bbonds_1b(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds):
    logger.info('Getting possible 1 breaking bond rearrangements')

    for bbond in all_possible_bbonds[0]:
        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[], bbonds=[bbond])

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds):
    logger.info('Getting possible 2 breaking bond rearrangements')

    if len(all_possible_bbonds) == 1:
        for bbond1, bbond2 in itertools.combinations(all_possible_bbonds[0], 2):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[], bbonds=[bbond1, bbond2])
    elif len(all_possible_bbonds) == 2:
        for bbond1, bbond2 in itertools.product(all_possible_bbonds[0], all_possible_bbonds[1]):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[], bbonds=[bbond1, bbond2])

    return possible_bond_rearrangs


def get_fbonds_bbonds_1b1f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds):
    logger.info('Getting possible 1 breaking and 1 forming bond rearrangements')

    if len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        for fbond, bbond in itertools.product(all_possible_fbonds[0], all_possible_bbonds[0]):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond])

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 0:
        for bbonds, fbonds in possible_fbond_and_bbonds:
            for bbond, fbond in itertools.product(bbonds, fbonds):
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond])

    return possible_bond_rearrangs


def get_fbonds_bbonds_1b2f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds):
    logger.info('Getting possible 1 breaking and 2 forming bond rearrangements')

    if len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 2:
        for fbond1, fbond2, bbond in itertools.product(all_possible_fbonds[0], all_possible_fbonds[1], all_possible_bbonds[0]):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        for bbond, (fbond1, fbond2) in itertools.product(all_possible_bbonds[0], itertools.combinations(all_possible_fbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond])

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 1:
        for bbonds, fbonds in possible_fbond_and_bbonds:
            for fbond1, fbond2, bbond in itertools.product(all_possible_fbonds[0], fbonds, bbonds):
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond])
        for bbond, (fbond1, fbond2) in itertools.product(fbond_atom_type_bbonds, itertools.combinations(all_possible_fbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond])

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b1f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds):
    logger.info('Getting possible 2 breaking and 1 forming bond rearrangements')

    if len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 1:
        for fbond, bbond1, bbond2 in itertools.product(all_possible_fbonds[0], all_possible_bbonds[0], all_possible_bbonds[1]):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        for fbond, (bbond1, bbond2) in itertools.product(all_possible_fbonds[0], itertools.combinations(all_possible_bbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 0:
        for bbonds, fbonds in possible_fbond_and_bbonds:
            for fbond, bbond1, bbond2 in itertools.product(fbonds, all_possible_bbonds[0], bbonds):
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond1, bbond2])
        for fbond, (bbond1, bbond2) in itertools.product(bbond_atom_type_fbonds, itertools.combinations(all_possible_bbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond1, bbond2])

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b2f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds):
    logger.info('Getting possible 2 breaking and 2 forming bond rearrangements')

    if len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 2:
        for fbond1, fbond2, bbond1, bbond2 in itertools.product(all_possible_fbonds[0], all_possible_fbonds[1], all_possible_bbonds[0], all_possible_bbonds[1]):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 1:
        for bbond1, bbond2, (fbond1, fbond2) in itertools.product(all_possible_bbonds[0], all_possible_bbonds[1], itertools.combinations(all_possible_fbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 2:
        for fbond1, fbond2, (bbond1, bbond2) in itertools.product(all_possible_fbonds[0], all_possible_fbonds[1], itertools.combinations(all_possible_bbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        for (fbond1, fbond2), (bbond1, bbond2) in itertools.product(itertools.combinations(all_possible_fbonds[0], 2), itertools.combinations(all_possible_bbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])
        for bbonds, fbonds in possible_fbond_and_bbonds:
            for fbond1, fbond2, bbond1, bbond2 in itertools.product(all_possible_fbonds[0], fbonds, all_possible_bbonds[0], bbonds):
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])
        for fbond1, fbond2, (bbond1, bbond2) in itertools.product(all_possible_fbonds[0], bbond_atom_type_fbonds, itertools.combinations(all_possible_bbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])
        for bbond1, bbond2, (fbond1, fbond2) in itertools.product(all_possible_bbonds[0], fbond_atom_type_bbonds, itertools.combinations(all_possible_fbonds[0], 2)):
            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 0:
        for m in range(len(possible_fbond_and_bbonds)):
            for n in range(len(possible_fbond_and_bbonds)):
                if m == n:
                    for o in range(len(possible_fbond_and_bbonds[m][0])):
                        for p in range(len(possible_fbond_and_bbonds[m][0])):
                            if p > o:
                                for q in range(len(possible_fbond_and_bbonds[m][1])):
                                    for r in range(len(possible_fbond_and_bbonds[m][1])):
                                        if q > r:
                                            bbond1, bbond2 = possible_fbond_and_bbonds[m][0][o], possible_fbond_and_bbonds[m][0][p]
                                            fbond1, fbond2 = possible_fbond_and_bbonds[m][1][q], possible_fbond_and_bbonds[m][1][r]
                                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

                elif m > n:
                    for bbond1 in possible_fbond_and_bbonds[m][0]:
                        for bbond2 in possible_fbond_and_bbonds[n][0]:
                            for fbond1 in possible_fbond_and_bbonds[m][1]:
                                for fbond2 in possible_fbond_and_bbonds[n][1]:
                                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

    return possible_bond_rearrangs


def strip_equivalent_bond_rearrangs(mol, possible_bond_rearrangs, depth=6):
    """Remove any bond rearrangement from possible_bond_rearrangs for which there is already an equivalent in the
    unique_bond_rearrangements list

    Arguments:
        mol (molecule object): reactant object
        possible_bond_rearrangs (list(object)): list of BondRearrangement objects

    Keyword Arguments:
        depth (int): Depth of neighbour list that must be identical for a set of atoms to be considered equivalent (default: {6})

    Returns:
        list(object): stripped list of BondRearrangement objects
    """
    logger.info('Stripping the forming and breaking bond list by discarding rearrangements with equivalent atoms')

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

    logger.info(f'Stripped {len(possible_bond_rearrangs)-len(unique_bond_rearrangements)} bond rearrangements')
    return unique_bond_rearrangements


class BondRearrangement:

    def __str__(self):
        return '_'.join(f'{bond[0]}-{bond[1]}' for bond in self.all)

    def get_active_atom_neighbour_lists(self, mol, depth):

        if self.active_atom_nl is None:
            self.active_atom_nl = [get_neighbour_list(species=mol, atom_i=atom)[:depth] for atom in self.active_atoms]

        return self.active_atom_nl

    def __eq__(self, other):
        return self.fbonds == other.fbonds and self.bbonds == other.bbonds

    def __init__(self, forming_bonds=None, breaking_bonds=None):

        self.fbonds = forming_bonds if forming_bonds is not None else []
        self.bbonds = breaking_bonds if breaking_bonds is not None else []

        self.n_fbonds = len(self.fbonds)
        self.n_bbonds = len(self.bbonds)

        self.all = self.fbonds + self.bbonds
        self.active_atoms = [atom_id for bond in self.all for atom_id in bond]
        self.active_atom_nl = None

        self.fatoms = [atom_id for bond in self.fbonds for atom_id in bond]
        self.batoms = [atom_id for bond in self.bbonds for atom_id in bond]
