import itertools


def gen_equiv_bond_rearrangs(identical_pairs, init_bond_rearrang):
    """
    Generate all possible combinations of equivelent BondRearrangement objects given an inital one and a list of
    identical pairs/bonds in a molecule. This function may return possibilities which are not chemically sensible,
    however it should find all the possibles, which realistic forming and breaking bond sets can be compared to

    :param identical_pairs: (dict) keyed with atom indexes and value as a list of tuples
    :param init_bond_rearrang: (object) BondRearrangement object
    :return:
    """

    fbonds_equivs = []
    for fbond in init_bond_rearrang.fbonds:
        fbonds_equivs.append(identical_pairs[fbond] + [fbond])
    fbonds_combinations = list(itertools.product(*fbonds_equivs))

    bbonds_equivs = []
    for bbond in init_bond_rearrang.bbonds:
        bbonds_equivs.append(identical_pairs[bbond] + [bbond])
    bbonds_combinations = list(itertools.product(*bbonds_equivs))

    bond_rearrangs = []
    for fbonds_bbonds in itertools.product(*(fbonds_combinations, bbonds_combinations)):
        fbonds, bbonds = fbonds_bbonds
        bond_rearrangs.append(BondRearrangement(forming_bonds=list(fbonds), breaking_bonds=list(bbonds)))

    return bond_rearrangs


class BondRearrangement(object):

    def __eq__(self, other):
        return self.fbonds == other.fbonds and self.bbonds == other.bbonds

    def __init__(self, forming_bonds=None, breaking_bonds=None):

        self.fbonds = forming_bonds if forming_bonds is not None else []
        self.bbonds = breaking_bonds if breaking_bonds is not None else []

        self.n_fbonds = len(self.fbonds)
        self.n_bbonds = len(self.bbonds)

        self.all = self.fbonds + self.bbonds
