from autode.geom import get_neighbour_list
from autode.log import logger
from autode import mol_graphs
from autode.atoms import get_maximal_valance
from autode.mol_graphs import is_isomorphic


def get_bond_rearrangs(mol, product):
    """For a reactant and product (complex) find the set of breaking and forming bonds that will turn reactants into
    products. This works by determining the types of bonds that have been made/broken (i.e CH) and then only considering
    rearrangements involving those bonds. 

    Arguments:
        mol {molecule object} -- reactant complex
        product {molecule object} -- product complex

    Returns:
        list -- list of bond rearrang objects linking reacs and prods
    """
    logger.info('Finding the possible forming and breaking bonds')

    if is_isomorphic(mol.graph, product.graph) and product.n_atoms > 3:
        logger.error('Reactant (complex) is isomorphic to product (complex). Bond rearrangement '
                     'cannot be determined unless the substrates are limited in size')
        return None

    possible_bond_rearrangements = []

    reac_bond_dict = mol_graphs.get_bond_type_list(mol.graph)
    prod_bond_dict = mol_graphs.get_bond_type_list(product.graph)

    all_possible_bbonds = []
    all_possible_fbonds = []
    possible_fbond_and_bbonds = []
    n_isomorphisms = 0
    fbond_atom_type_bbonds = None
    bbond_atom_type_fbonds = None

    for reac_key, reac_bonds in reac_bond_dict.items():
        prod_bonds = prod_bond_dict[reac_key]
        possible_fbonds = mol_graphs.get_fbonds(mol.graph, reac_key)
        if len(prod_bonds) < len(reac_bonds):
            all_possible_bbonds.append(reac_bonds)
            bbond_atom_type_fbonds = possible_fbonds
        elif len(prod_bonds) > len(reac_bonds):
            all_possible_fbonds.append(possible_fbonds)
            fbond_atom_type_bbonds = reac_bonds
        else:
            if len(reac_bonds) != 0:
                possible_fbond_and_bbonds.append([reac_bonds, possible_fbonds])
                n_isomorphisms += len(possible_fbonds) * len(reac_bonds)

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
        logger.error(f'Cannot treat a change in bonds reactant <- product of {delta_n_bonds}')
        return None

    for func in funcs:
        possible_bond_rearrangements = func(mol, product, possible_bond_rearrangements, all_possible_bbonds, all_possible_fbonds,
                                            possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms)
        if len(possible_bond_rearrangements) > 0:
            logger.info(f'Found a molecular graph rearrangement to products with {func.__name__}')
            # This function will return with from the first bond rearrangement that leads to products

            n_bond_rearrangs = len(possible_bond_rearrangements)
            if n_bond_rearrangs > 1:
                logger.info(f'Multiple *{n_bond_rearrangs}* possible bond breaking/makings are possible')
                possible_bond_rearrangements = strip_equivalent_bond_rearrangs(mol, possible_bond_rearrangements)

            return possible_bond_rearrangements

    return None


def add_bond_rearrangment(bond_rearrangs, reactant, product, fbonds, bbonds):
    """For a possible bond rearrangement, sees if the products are made, and adds it to the bond rearrang list if it does

    Arguments:
        bond_rearrangs {list} -- list of working bond rearrangments
        reactant {molecule object} -- reactant complex
        product {molecule object} -- product complex
        fbonds {list of tuples} -- list of bonds to be made
        bbonds {list of tuples} -- list of bonds to be broken

    Returns:
        list -- updated list of working bond rearrangments
    """

    # Check that the bond rearrangement doesn't exceed standard atom valances
    bbond_atoms = [atom for bbond in bbonds for atom in bbond]
    for fbond in fbonds:
        for atom in fbond:
            atom_label = reactant.get_atom_label(atom)
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
        graph {nx.Graph} -- reactant graph
        fbonds {list(tuple)} -- list of bonds to be made
        bbonds {list(tuple)} -- list of bonds to be broken

    Returns:
        {nx.Graph} -- rearranged graph
    """
    rearranged_graph = graph.copy()
    for fbond in fbonds:
        rearranged_graph.add_edge(*fbond)
    for bbond in bbonds:
        rearranged_graph.remove_edge(*bbond)

    return rearranged_graph


def get_fbonds_bbonds_1b(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms):
    logger.info('Getting possible 1 breaking bond rearrangements')

    possible_bbonds = all_possible_bbonds[0]

    logger.info(f'Have {len(possible_bbonds)} isomorphisms to do')

    for bbond in possible_bbonds:
        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                        fbonds=[], bbonds=[bbond])
    return possible_bond_rearrangs


def get_fbonds_bbonds_2b(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms):
    logger.info('Getting possible 2 breaking bond rearrangements')

    if len(all_possible_bbonds) == 1:
        possible_bbonds = all_possible_bbonds[0]
        logger.info(f'Have {int(0.5*len(possible_bbonds)*(len(possible_bbonds)-1))} isomorphisms to do')
        for i in range(len(possible_bbonds)):
            for j in range(len(possible_bbonds)):
                if i > j:
                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[], bbonds=[possible_bbonds[i], possible_bbonds[j]])
    elif len(all_possible_bbonds) == 2:
        possible_bbonds1 = all_possible_bbonds[0]
        possible_bbonds2 = all_possible_bbonds[1]
        logger.info(f'Have {len(possible_bbonds1)*len(possible_bbonds2)} isomorphisms to do')
        for bbond1 in possible_bbonds1:
            for bbond2 in possible_bbonds2:
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[], bbonds=[bbond1, bbond2])

    return possible_bond_rearrangs


def get_fbonds_bbonds_1b1f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms):
    logger.info('Getting possible 1 breaking and 1 forming bond rearrangements')

    if len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        possible_bbonds = all_possible_bbonds[0]
        possible_fbonds = all_possible_fbonds[0]
        logger.info(f'Have {len(possible_bbonds)*len(possible_fbonds)} isomorphisms to do')

        for fbond in possible_fbonds:
            for bbond in possible_bbonds:
                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product,
                                                                fbonds=[fbond], bbonds=[bbond])

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 0:
        logger.info(f'Have {n_isomorphisms} isomorphisms to do')
        for bbonds, fbonds in possible_fbond_and_bbonds:
            for bbond in bbonds:
                for fbond in fbonds:
                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond])

    return possible_bond_rearrangs


def get_fbonds_bbonds_1b2f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms):
    logger.info('Getting possible 1 breaking and 2 forming bond rearrangements')

    if len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 2:
        possible_bbonds = all_possible_bbonds[0]
        possible_fbonds1 = all_possible_fbonds[0]
        possible_fbonds2 = all_possible_fbonds[1]
        logger.info(f'Have {len(possible_bbonds)*len(possible_fbonds1)*len(possible_fbonds2)} isomorphisms to do')
        for fbond1 in possible_fbonds1:
            for fbond2 in possible_fbonds2:
                for bbond in possible_bbonds:
                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        possible_bbonds = all_possible_bbonds[0]
        possible_fbonds = all_possible_fbonds[0]
        logger.info(f'Have {int(len(possible_bbonds)*0.5*len(possible_fbonds)*(len(possible_fbonds)-1))} isomorphisms to do')
        for bbond in possible_bbonds:
            for i in range(len(possible_fbonds)):
                for j in range(len(possible_fbonds)):
                    if i > j:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[possible_fbonds[i], possible_fbonds[j]], bbonds=[bbond])

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 1:
        possible_fbonds = all_possible_fbonds[0]
        isomorphisms_to_do = (len(fbond_atom_type_bbonds)*0.5*len(possible_fbonds)*(len(possible_fbonds)-1)) + (len(possible_fbonds * n_isomorphisms))
        logger.info(f'Have {int(isomorphisms_to_do)} isomorphisms to do')
        for fbond1 in possible_fbonds:
            for bbonds, fbonds in possible_fbond_and_bbonds:
                for bbond in bbonds:
                    for fbond2 in fbonds:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond])
        for i in range(len(possible_fbonds)):
            for j in range(len(possible_fbonds)):
                if i > j:
                    for bbond in fbond_atom_type_bbonds:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[possible_fbonds[i], possible_fbonds[j]], bbonds=[bbond])

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b1f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms):
    logger.info('Getting possible 2 breaking and 1 forming bond rearrangements')

    if len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 1:
        possible_fbonds = all_possible_fbonds[0]
        possible_bbonds1 = all_possible_bbonds[0]
        possible_bbonds2 = all_possible_bbonds[1]
        logger.info(f'Have {len(possible_fbonds)*len(possible_bbonds1)*len(possible_bbonds2)} isomorphisms to do')
        for bbond1 in possible_bbonds1:
            for bbond2 in possible_bbonds2:
                for fbond in possible_fbonds:
                    possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        possible_bbonds = all_possible_bbonds[0]
        possible_fbonds = all_possible_fbonds[0]
        logger.info(f'Have {int(len(possible_fbonds)*0.5*len(possible_bbonds)*(len(possible_bbonds)-1))} isomorphisms to do')
        for fbond in possible_fbonds:
            for i in range(len(possible_bbonds)):
                for j in range(len(possible_bbonds)):
                    if i > j:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[possible_bbonds[i], possible_bbonds[j]])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 0:
        possible_bbonds = all_possible_bbonds[0]
        isomorphisms_to_do = (len(bbond_atom_type_fbonds)*0.5*len(possible_bbonds)*(len(possible_bbonds)-1)) + (len(possible_bbonds * n_isomorphisms))
        logger.info(f'Have {int(isomorphisms_to_do)} isomorphisms to do')
        for bbond1 in possible_bbonds:
            for bbonds, fbonds in possible_fbond_and_bbonds:
                for bbond2 in bbonds:
                    for fbond in fbonds:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[bbond1, bbond2])
        for i in range(len(possible_bbonds)):
            for j in range(len(possible_bbonds)):
                if i > j:
                    for fbond in bbond_atom_type_fbonds:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond], bbonds=[possible_bbonds[i], possible_bbonds[j]])

    return possible_bond_rearrangs


def get_fbonds_bbonds_2b2f(reactant, product, possible_bond_rearrangs, all_possible_bbonds, all_possible_fbonds, possible_fbond_and_bbonds, bbond_atom_type_fbonds, fbond_atom_type_bbonds, n_isomorphisms):
    logger.info('Getting possible 2 breaking and 2 forming bond rearrangements')

    if len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 2:
        possible_fbonds1 = all_possible_fbonds[0]
        possible_fbonds2 = all_possible_fbonds[1]
        possible_bbonds1 = all_possible_bbonds[0]
        possible_bbonds2 = all_possible_bbonds[1]
        logger.info(f'Have {len(possible_fbonds1)*len(possible_fbonds2)*len(possible_bbonds1)*len(possible_bbonds2)} isomorphisms to do')
        for bbond1 in possible_bbonds1:
            for bbond2 in possible_bbonds2:
                for fbond1 in possible_fbonds1:
                    for fbond2 in possible_fbonds2:
                        possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 1:
        possible_bbonds1 = all_possible_bbonds[0]
        possible_bbonds2 = all_possible_bbonds[1]
        possible_fbonds = all_possible_fbonds[0]
        logger.info(f'Have {int(len(possible_bbonds1)*len(possible_bbonds2)*0.5*len(possible_fbonds)*(len(possible_fbonds)-1))} isomorphisms to do')
        for bbond1 in possible_bbonds1:
            for bbond2 in possible_bbonds2:
                for i in range(len(possible_fbonds)):
                    for j in range(len(possible_fbonds)):
                        if i > j:
                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[possible_fbonds[i], possible_fbonds[j]], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 2:
        possible_bbonds = all_possible_bbonds[0]
        possible_fbonds1 = all_possible_fbonds[0]
        possible_fbonds2 = all_possible_fbonds[1]
        logger.info(f'Have {int(len(possible_fbonds1)*len(possible_fbonds2)*0.5*len(possible_bbonds)*(len(possible_bbonds)-1))} isomorphisms to do')
        for fbond1 in possible_fbonds1:
            for fbond2 in possible_fbonds2:
                for i in range(len(possible_bbonds)):
                    for j in range(len(possible_bbonds)):
                        if i > j:
                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[possible_bbonds[i], possible_bbonds[j]])

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        possible_bbonds = all_possible_bbonds[0]
        possible_fbonds = all_possible_fbonds[0]
        isomorphisms_to_do = (0.5*len(possible_fbonds)*(len(possible_fbonds)-1)*0.5*len(possible_bbonds)*(len(possible_bbonds)-1)) + (len(possible_bbonds) * len(possible_fbonds) * n_isomorphisms)
        logger.info(f'Have {int(isomorphisms_to_do)} isomorphisms to do')
        for m in range(len(possible_fbonds)):
            for n in range(len(possible_fbonds)):
                if m > n:
                    for i in range(len(possible_bbonds)):
                        for j in range(len(possible_bbonds)):
                            if i > j:
                                bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                                fbond1, fbond2 = possible_fbonds[m], possible_fbonds[n]
                                possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])
        for bbond1 in possible_bbonds:
            for fbond1 in possible_fbonds:
                for bbonds, fbonds in possible_fbond_and_bbonds:
                    for bbond2 in bbonds:
                        for fbond2 in fbonds:
                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[bbond1, bbond2])
        for fbond1 in possible_fbonds:
            for i in range(len(possible_bbonds)):
                for j in range(len(possible_bbonds)):
                    if i > j:
                        for fbond2 in bbond_atom_type_fbonds:
                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[fbond1, fbond2], bbonds=[possible_bbonds[i], possible_bbonds[j]])
        for bbond1 in possible_bbonds:
            for i in range(len(possible_fbonds)):
                for j in range(len(possible_fbonds)):
                    if i > j:
                        for bbond2 in fbond_atom_type_bbonds:
                            possible_bond_rearrangs = add_bond_rearrangment(possible_bond_rearrangs, reactant, product, fbonds=[possible_bbonds[i], possible_bbonds[j]], bbonds=[bbond1, bbond2])

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 0:
        logger.info('Have lots of isomorphisms to do')
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
        mol {molecule object} -- reactant object
        possible_bond_rearrangs {list(object)} -- list of BondRearrangement objects

    Keyword Arguments:
        depth {int} -- Depth of neighbour list that must be identical for a set of atoms to be considered equivalent (default: {6})

    Returns:
        {list(object)} -- stripped list of BondRearrangement objects
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


class BondRearrangement(object):

    def get_active_atom_neighbour_lists(self, mol, depth):

        if self.active_atom_nl is None:
            self.active_atom_nl = [get_neighbour_list(atom_i=atom, mol=mol)[:depth] for atom in self.active_atoms]

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
