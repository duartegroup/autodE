from autode.transition_states import locate_tss
from autode.atoms import Atom
from autode.molecule import Reactant, Product
from autode.transition_states import locate_tss
from autode.reaction import Reaction
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.templates import get_ts_templates
import autode.bond_rearrangement as bond_rearrangement

# H2 -> H + H dissociation
h_product_1 = Product(name='h_a', atoms=[Atom('H', 0.0, 0.0, 0.0)])
h_product_2 = Product(name='h_b', atoms=[Atom('H', 0.0, 0.0, 0.0)])

hh_reactant = Reactant(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('H', 0.7, 0.0, 0.0)])

dissoc_reaction = Reaction(name='dissoc', mol1=h_product_1, mol2=h_product_2, mol3=hh_reactant)


# H2 + h > h + mol substitution
h_reactant = Reactant(name='H_c', atoms=[Atom('H', 0.0, 0.0, 0.0)])
hh_product = Product(name='H2', atoms=[Atom('H', 0.7, 0.0, 0.0), Atom('H', 1.4, 0.0, 0.0)])

subs_reaction = Reaction(name='subs', mol1=h_reactant, mol2=hh_reactant, mol3=hh_product, mol4=h_product_2)


def test_find_tss():

    tss = locate_tss.find_tss(dissoc_reaction)

    assert len(tss) == 1
