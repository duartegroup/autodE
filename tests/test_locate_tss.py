from autode.transition_states import locate_tss
from autode import molecule
from autode import reaction
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.template_ts_guess import get_template_ts_guess

# h + h > h2 dissociation
h_product_1 = molecule.Product(name='H', xyzs=[['H', 0.0, 0.0, 0.0]])
h_product_2 = molecule.Product(name='H', xyzs=[['H', 1.0, 0.0, 0.0]])
hh_reactant = molecule.Reactant(
    name='H2', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
dissoc_reaction = reaction.Reaction(
    name='dissoc', mol1=h_product_1, mol2=h_product_2, mol3=hh_reactant)
dissoc_reactant, dissoc_product = locate_tss.get_reactant_and_product_complexes(
    dissoc_reaction)
dissoc_rearrangs = locate_tss.get_bond_rearrangs(
    dissoc_reactant, dissoc_product)

# h2 + h > h + h2 substitution
h_reactant = molecule.Reactant(name='H', xyzs=[['H', 0.0, 0.0, 0.0]])
hh_product = molecule.Product(
    name='H2', xyzs=[['H', 0.7, 0.0, 0.0], ['H', 1.4, 0.0, 0.0]])
subs_reaction = reaction.Reaction(
    name='subs', mol1=h_reactant, mol2=hh_reactant, mol3=hh_product, mol4=h_product_2)
subs_reactant, subs_product = locate_tss.get_reactant_and_product_complexes(
    subs_reaction)
subs_rearrangs = locate_tss.get_bond_rearrangs(subs_reactant, subs_product)

# ch2fch2+ > +chfch3 rearrangement
alkyl_reactant = molecule.Reactant(smiles='[CH2+]CF')
alkyl_product = molecule.Product(smiles='C[CH+]F')
rearang_reaction = reaction.Reaction(
    name='rearang', mol1=alkyl_reactant, mol2=alkyl_product)
rearrang_reactant, rearrang_product = locate_tss.get_reactant_and_product_complexes(
    rearang_reaction)
rearrang_rearrangs = locate_tss.get_bond_rearrangs(
    rearrang_reactant, rearrang_product)


def test_get_reac_and_prod_complexes():

    assert type(dissoc_reactant) == molecule.Reactant
    assert type(dissoc_product) == molecule.Molecule
    assert len(dissoc_reactant.xyzs) == 2
    assert len(dissoc_product.xyzs) == 2
    assert dissoc_product.xyzs[1][3] - dissoc_product.xyzs[0][3] > 90

    assert type(subs_reactant) == molecule.Molecule
    assert type(subs_product) == molecule.Molecule
    assert len(subs_reactant.xyzs) == 3
    assert len(subs_product.xyzs) == 3

    assert type(rearrang_reactant) == molecule.Reactant
    assert type(rearrang_product) == molecule.Product
    assert len(rearrang_reactant.xyzs) == 7
    assert len(rearrang_product.xyzs) == 7


def test_get_bond_rearrangs():

    assert len(dissoc_rearrangs) == 1
    assert type(dissoc_rearrangs[0]) == BondRearrangement
    assert dissoc_rearrangs[0].all == [(0, 1)]

    assert len(subs_rearrangs) == 1
    assert type(subs_rearrangs[0]) == BondRearrangement
    assert subs_rearrangs[0].all == [(0, 1), (1, 2)]

    assert len(rearrang_rearrangs) == 1
    assert type(rearrang_rearrangs[0]) == BondRearrangement
    assert rearrang_rearrangs[0].all == [(0, 5), (1, 5)]


def test_add_bond_rearrangement():
    # rearrangment doesn't get product, shouldn't be added
    bond_rearrangs = locate_tss.add_bond_rearrangment(
        [], subs_reactant, subs_product, [(0, 1)], [])
    assert bond_rearrangs == []


def test_rearranged_graph():
    rearranged_graph = locate_tss.generate_rearranged_graph(
        subs_reactant.graph, [(0, 1)], [(1, 2)])
    assert locate_tss.mol_graphs.is_isomorphic(
        rearranged_graph, subs_product.graph)


def test_get_funcs_and_params():
    dissoc_rearrang = dissoc_rearrangs[0]
    dissoc_funcs_params = [
        (get_template_ts_guess, (dissoc_reactant, dissoc_rearrang.all, dissoc_reaction.type))]
    dissoc_f_and_p = locate_tss.get_ts_guess_funcs_and_params(dissoc_funcs_params,
                                                              dissoc_reaction, dissoc_reactant, dissoc_product, dissoc_rearrang)
    assert type(dissoc_f_and_p) == list
    assert len(dissoc_f_and_p) == 4
    assert dissoc_f_and_p[0][0] == locate_tss.get_template_ts_guess
    assert type(dissoc_f_and_p[1][1][0]) == molecule.Reactant
    assert dissoc_f_and_p[1][1][2] == (0, 1)
    assert dissoc_f_and_p[1][1][4] == 'H2--H+H_0-1_ll1d'
    assert dissoc_f_and_p[3][1][5] == locate_tss.Dissociation

    subs_rearrang = subs_rearrangs[0]
    subs_funcs_params = [
        (get_template_ts_guess, (subs_reactant, subs_rearrang.all, subs_reaction.type))]
    subs_f_and_p = locate_tss.get_ts_guess_funcs_and_params(
        subs_funcs_params, subs_reaction, subs_reactant, subs_product, subs_rearrangs[0])
    assert type(subs_f_and_p) == list
    assert len(subs_f_and_p) == 9
    assert subs_f_and_p[3][1][5] == locate_tss.Substitution


def test_strip_equivalent_rearrangements():
    # 5 and 6 should be equivalent Hs
    possible_bond_rearrangs = [BondRearrangement(
        [(0, 5)], [(1, 5)]), BondRearrangement([(0, 6)], [(1, 6)])]
    unique_rearrangs = locate_tss.strip_equivalent_bond_rearrangs(
        rearrang_reactant, possible_bond_rearrangs)
    assert len(unique_rearrangs) == 1


def test_get_fbonds_bbonds():
    reac1 = molecule.Molecule(
        xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.6], ['H', 0.0, 0.0, 1.2]])
    prod1 = molecule.Molecule(
        xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 50], ['H', 0.0, 0.0, 100]])
    two_bbond = locate_tss.get_fbonds_bbonds_2b(reac1, prod1, [])
    assert two_bbond == [BondRearrangement(breaking_bonds=[(0, 1), (1, 2)])]

    reac2 = molecule.Molecule(
        xyzs=[['H', 0.0, 0.0, -1.2], ['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.6], ['H', 0.0, 0.0, 1.2]])
    prod2 = molecule.Molecule(
        xyzs=[['H', 0.0, 0.0, -0.6], ['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 5], ['H', 0.0, 0.0, 10]])
    two_bbond_one_fbond = locate_tss.get_fbonds_bbonds_2b1f(reac2, prod2, [])
    assert two_bbond_one_fbond == [BondRearrangement(forming_bonds=[(0, 1)], breaking_bonds=[(1, 2), (2, 3)]),
                                    BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(1, 2), (2, 3)]),
                                    BondRearrangement(forming_bonds=[(0, 3)], breaking_bonds=[(1, 2), (2, 3)]),
                                    BondRearrangement(forming_bonds=[(1, 3)], breaking_bonds=[(1, 2), (2, 3)])]

    one_bbond_two_fbond = locate_tss.get_fbonds_bbonds_1b2f(prod2, reac2, [])
    assert one_bbond_two_fbond == [BondRearrangement(forming_bonds=[(0, 2), (0, 3)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(0, 2), (1, 2)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(0, 3), (1, 3)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(1, 2), (1, 3)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(0, 2), (2, 3)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(1, 2), (2, 3)], breaking_bonds=[(0, 1)]),
                                    BondRearrangement(forming_bonds=[(1, 3), (2, 3)], breaking_bonds=[(0, 1)])]

    reac3 = molecule.Molecule(xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.6], [
                              'H', 10, 0.0, 0.0], ['H', 10, 0.0, 0.6]])
    prod3 = molecule.Molecule(xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 10], [
                              'H', 0.6, 0.0, 0.0], ['H', 0.6, 0.0, 10]])
    two_bbond_two_fbond = locate_tss.get_fbonds_bbonds_2b2f(reac3, prod3, [])
    assert two_bbond_two_fbond == [BondRearrangement(forming_bonds=[(0, 3), (1, 2)], breaking_bonds=[(0, 1), (2, 3)]), BondRearrangement(forming_bonds=[(0, 2), (1, 3)], breaking_bonds=[(0, 1), (2, 3)])]
