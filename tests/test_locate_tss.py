from autode.transition_states import locate_tss
from autode.atoms import Atom
from autode import molecule
from autode import reaction
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.template_ts_guess import get_template_ts_guess
import autode.bond_rearrangement as bond_rearrangement

# h + h > mol dissociation
h_product_1 = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)])
h_product_2 = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)])

hh_reactant = reaction.Reactant(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)])
dissoc_reaction = reaction.Reaction(name='dissoc', mol1=h_product_1, mol2=h_product_2, mol3=hh_reactant)
for mol in dissoc_reaction.reacs + dissoc_reaction.prods:
    mol.charges = []
dissoc_reactant, dissoc_product = locate_tss.get_reactant_and_product_complexes(dissoc_reaction)
dissoc_rearrangs = locate_tss.get_bond_rearrangs(dissoc_reactant, dissoc_product)

# mol + h > h + mol substitution
h_reactant = molecule.Reactant(name='H', atoms=[Atom('H', 0.0, 0.0, 0.0)])
hh_product = molecule.Product(name='H2', atoms=[Atom('H', 0.7, 0.0, 0.0), Atom('H', 1.4, 0.0, 0.0)])
subs_reaction = reaction.Reaction(name='subs', mol1=h_reactant, mol2=hh_reactant, mol3=hh_product, mol4=h_product_2)
for mol in subs_reaction.reacs + subs_reaction.prods:
    mol.charges = []
subs_reactant, subs_product = locate_tss.get_reactant_and_product_complexes(subs_reaction)
subs_rearrangs = locate_tss.get_bond_rearrangs(subs_reactant, subs_product)


def test_get_funcs_and_params():
    dissoc_f_and_p = locate_tss.get_ts_guess_funcs_and_params([], dissoc_reaction, dissoc_reactant, dissoc_product, dissoc_rearrangs[0], None)
    assert type(dissoc_f_and_p) == list
    assert len(dissoc_f_and_p) == 3
    assert type(dissoc_f_and_p[0][1][0]) == molecule.Reactant
    assert dissoc_f_and_p[0][1][2] == (0, 1)
    assert dissoc_f_and_p[0][1][4] == 'H2--H+H_0-1_ll1d'
    assert dissoc_f_and_p[2][1][5] == locate_tss.Dissociation

    subs_f_and_p = locate_tss.get_ts_guess_funcs_and_params([], subs_reaction, subs_reactant, subs_product, subs_rearrangs[0], None)
    assert type(subs_f_and_p) == list
    assert len(subs_f_and_p) == 8
    assert subs_f_and_p[2][1][5] == locate_tss.Substitution

    butadiene = molecule.Reactant(name='butadiene', smiles='C=CC=C')
    ethene = molecule.Reactant(name='ethene', smiles='C=C')
    cyclohexene = molecule.Product(name='cyclohexene', smiles='C1C=CCCC1')
    d_a = reaction.Reaction(butadiene, ethene, cyclohexene)
    for mol in d_a.reacs + d_a.prods:
        mol.charges = []
    d_a_reactant, d_a_product = locate_tss.get_reactant_and_product_complexes(d_a)
    print(d_a_reactant.graph.edges)
    print(d_a_product.graph.edges)
    d_a_rearrangs = locate_tss.get_bond_rearrangs(d_a_reactant, d_a_product)
    d_a_f_and_p = locate_tss.get_ts_guess_funcs_and_params([], d_a, d_a_reactant, d_a_product, d_a_rearrangs[0], None)
    assert type(d_a_f_and_p) == list
    assert len(d_a_f_and_p) == 2
    assert d_a_f_and_p[0][0] == locate_tss.get_ts_guess_2d
    assert d_a_f_and_p[1][1][5] == 'cyclohexene--butadiene+ethene_0-5_3-4_hl2d_bbonds'


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

    # ch2fch2+ > +chfch3 rearrangement
    alkyl_reactant = molecule.Reactant(smiles='[CH2+]CF')
    alkyl_product = molecule.Product(smiles='C[CH+]F')
    rearang_reaction = reaction.Reaction(name='rearang', mol1=alkyl_reactant, mol2=alkyl_product)
    for mol in rearang_reaction.reacs + rearang_reaction.prods:
        mol.charges = []
    rearrang_reactant, rearrang_product = locate_tss.get_reactant_and_product_complexes(rearang_reaction)
    assert type(rearrang_reactant) == molecule.Reactant
    assert type(rearrang_product) == molecule.Product
    assert len(rearrang_reactant.xyzs) == 7
    assert len(rearrang_product.xyzs) == 7

    # h + ch3ch2cl > mol + ch2ch2 + Cl
    propyl_chloride = molecule.Reactant(smiles='CCCl')
    chlorine = molecule.Product(xyzs=[['Cl', 0.0, 0.0, 0.0]])
    ethene = molecule.Product(smiles='C=C')
    propyl_chloride.charges = [0, 0, 0, 0, 0]
    elim_reaction = reaction.Reaction(h_reactant, propyl_chloride, hh_product, ethene, chlorine)
    for mol in elim_reaction.reacs + elim_reaction.prods:
        mol.charges = []
    elim_reactant, elim_product = locate_tss.get_reactant_and_product_complexes(elim_reaction)
    assert type(elim_reactant) == molecule.Molecule
    assert type(elim_product) == molecule.Molecule
    assert len(elim_reactant.xyzs) == 9
    assert len(elim_product.xyzs) == 9
