from .log import logger
from .molecule import Molecule
from .reactions import Dissociation, Rearrangement, Substitution
from .mol_graphs import is_isomorphic
from .mol_graphs import get_adjacency_graph


def find_tss(reaction):
    logger.info('Finding possible transition states')
    # TODO elimination reactions
    tss = []
    reactant, product = get_reactant_and_product_complexes(reaction)
    fbonds_and_bbonds = get_forming_and_breaking_bonds(reactant, product)

    if len(fbonds_and_bbonds) > 1:
        logger.info('Multiple possible bond breaking/makings are possible')
        fbonds_and_bbonds = strip_identical_fbond_bbond_sets(reactant, fbonds_and_bbonds)


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


def get_forming_and_breaking_bonds(reactant, product):
    """
    For a reactant and product (complex) find the set of breaking and forming bonds that will turn reactants into
    products. Will run O(n), O(n^2), O(n^2), O(n^3) number of molecular graph isomorphisms. This could be slow...
    :param reactant: (object) Molecule object
    :param product: (object) Molecule object
    :return:
    """
    logger.info('Finding the possible forming and breaking bonds')

    fbonds_and_bbonds = []

    possible_fbonds = [(i, j) for i in range(reactant.n_atoms) for j in range(reactant.n_atoms) if i < j]
    possible_bbonds = [pair for pair in reactant.graph.edges() if sorted(pair) not in possible_fbonds]

    for func in [get_fbonds_bbonds_1b, get_fbonds_bbonds_2b, get_fbonds_bbonds_1b1f, get_fbonds_bbonds_2b1f]:
        fbonds_and_bbonds = func(possible_fbonds, possible_bbonds, reactant, product, fbonds_and_bbonds)
        if len(fbonds_and_bbonds) > 0:
            logger.info('Found a molecular graph rearrangement to products with {}'.format(func.__name__))
            return fbonds_and_bbonds

    logger.error('Could not find a set of forming/breaking bonds')
    return None


def get_fbonds_bbonds_1b(possible_fbonds, possible_bbonds, reactant, product, fbonds_and_bbonds):

    for bbond in possible_bbonds:
        rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[], bbonds=[bbond])
        if is_isomorphic(rearranged_graph, product.graph):
            fbonds_and_bbonds.append({None: (bbond,)})

    return fbonds_and_bbonds


def get_fbonds_bbonds_2b(possible_fbonds, possible_bbonds, reactant, product, fbonds_and_bbonds):

    for i in range(len(possible_bbonds)):
        for j in range(len(possible_bbonds)):
            if i > j:
                bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[], bbonds=[bbond1, bbond2])
                if is_isomorphic(rearranged_graph, product.graph):
                    fbonds_and_bbonds.append({None: (bbond1, bbond2)})

    return fbonds_and_bbonds


def get_fbonds_bbonds_1b1f(possible_fbonds, possible_bbonds, reactant, product, fbonds_and_bbonds):

    for fbond in possible_fbonds:
        for bbond in possible_bbonds:
            rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[fbond], bbonds=[bbond])
            if is_isomorphic(rearranged_graph, product.graph):
                fbonds_and_bbonds.append({(fbond,): (bbond,)})

    return fbonds_and_bbonds


def get_fbonds_bbonds_2b1f(possible_fbonds, possible_bbonds, reactant, product, fbonds_and_bbonds):

    for fbond in possible_fbonds:
        for i in range(len(possible_bbonds)):
            for j in range(len(possible_bbonds)):
                if i > j:
                    bbond1, bbond2 = possible_bbonds[i], possible_bbonds[j]
                    rearranged_graph = generate_rearranged_graph(reactant.graph, fbonds=[fbond],
                                                                 bbonds=[bbond1, bbond2])

                    if is_isomorphic(rearranged_graph, product.graph):
                        fbonds_and_bbonds.append({(fbond,): (bbond1, bbond2)})

    return fbonds_and_bbonds


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


def strip_identical_fbond_bbond_sets(reactant, fbonds_and_bbonds):
    logger.info('Stripping the forming and breaking bond list by discarding symmetry equivs')

    unique_fbond_bbonds = [fbonds_and_bbonds[0]]                                # First must be unique
    unique_atoms_and_adj_graph = {0: get_adjacency_graph(atom_i=0, graph=reactant.graph)}

    print(unique_atoms_and_adj_graph[0].nodes.data())

    for i in range(reactant.n_atoms):

        # TODO this function

        pass


    return unique_fbond_bbonds
