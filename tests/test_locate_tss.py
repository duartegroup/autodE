from autode.transition_states import locate_tss
from autode import molecule
from autode import reaction
from autode.bond_rearrangement import BondRearrangement

# h + h > h2 dissociation
h_product_1 = molecule.Product(xyzs=[['H', 0.0, 0.0, 0.0]])
h_product_2 = molecule.Product(xyzs=[['H', 1.0, 0.0, 0.0]])
hh_reactant = molecule.Reactant(xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
dissoc_reaction = reaction.Reaction(mol1=h_product_1, mol2=h_product_2, mol3=hh_reactant)
dissoc_reactant, dissoc_product = locate_tss.get_reactant_and_product_complexes(dissoc_reaction)
dissoc_rearrangs = locate_tss.get_bond_rearrangs(dissoc_reactant, dissoc_product)

# h2 + h > h + h2 substitution
h_reactant = molecule.Reactant(xyzs=[['H', 0.0, 0.0, 0.0]])
hh_product = molecule.Product(xyzs=[['H', 0.7, 0.0, 0.0], ['H', 1.4, 0.0, 0.0]])
subs_reaction = reaction.Reaction(mol1=h_reactant, mol2=hh_reactant, mol3=hh_product, mol4=h_product_2)
subs_reactant, subs_product = locate_tss.get_reactant_and_product_complexes(subs_reaction)
subs_rearrangs = locate_tss.get_bond_rearrangs(subs_reactant, subs_product)

def test_reac_and_prod_complexes():

    assert type(dissoc_reactant) == molecule.Reactant
    assert type(dissoc_product) == molecule.Molecule
    assert len(dissoc_reactant.xyzs) == 2
    assert len(dissoc_product.xyzs) == 2

    assert type(subs_reactant) == molecule.Molecule
    assert type(subs_product) == molecule.Molecule
    assert len(subs_reactant.xyzs) == 3
    assert len(subs_product.xyzs) == 3


def test_get_bond_rearrangs():
    
    assert len(dissoc_rearrangs) == 1
    assert type(dissoc_rearrangs[0]) == BondRearrangement
    assert dissoc_rearrangs[0].all == [(0,1)]

    assert len(subs_rearrangs) == 1
    assert type(subs_rearrangs[0]) == BondRearrangement
    assert subs_rearrangs[0].all == [(0,1), (1,2)]