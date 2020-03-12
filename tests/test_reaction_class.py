from autode import reaction
from autode.transition_states.transition_state import TS
from autode.atoms import Atom
from autode.exceptions import UnbalancedReaction
from autode.exceptions import SolventsDontMatch
import pytest

h1 = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)])

h2 = reaction.Reactant(name='h2', atoms=[Atom('H', 1.0, 0.0, 0.0)])
h2_product = reaction.Product(name='h2', atoms=[Atom('H', 1.0, 0.0, 0.0)])

hh_product = reaction.Product(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)])
hh_reactant = reaction.Reactant(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)])

lin_h3 = reaction.Reactant(name='h3_linear', atoms=[Atom('H', -1.76172, 0.79084, -0.00832),
                                                    Atom('H', -2.13052, 0.18085, 0.00494),
                                                    Atom('H', -1.39867, 1.39880, -0.00676)])

trig_h3 = reaction.Product(name='h3_trigonal', atoms=[Atom('H', -1.76172, 0.79084, -0.00832),
                                                      Atom('H', -1.65980, 1.15506, 0.61469),
                                                      Atom('H', -1.39867, 1.39880, -0.00676)])


def test_reaction_class():
    # h + h > mol
    hh_reac = reaction.Reaction(mol1=h1, mol2=h2, mol3=hh_product, name='h2_assoc')
    hh_reac.solvent_sphere_energy = 0

    h1.energy = 2
    h2.energy = 3
    hh_product.energy = 1

    assert hh_reac.type == reaction.reactions.Dissociation
    assert len(hh_reac.prods) == 2
    assert len(hh_reac.reacs) == 1
    assert hh_reac.ts is None
    assert hh_reac.tss == []
    assert hh_reac.name == 'h2_assoc'
    assert hh_reac.calc_delta_e() == 4

    # h + mol > mol + h
    h_sub = reaction.Reaction(mol1=h1, mol2=hh_reactant, mol3=h2_product, mol4=hh_product, solvent_name='water')

    assert h_sub.type == reaction.reactions.Substitution
    assert h_sub.name == 'reaction'
    assert h_sub.solvent.name == 'water'
    assert h_sub.solvent.smiles == 'O'


def test_solvated_reaction_class():

    hh_reac = reaction.SolvatedReaction(mol1=h1, mol2=h2, mol3=hh_product, name='h2_assoc')

    assert hh_reac.solvent_sphere_energy is None


def test_check_rearrangement():
    reac = reaction.Reaction(trig_h3, lin_h3)

    # Should switch reactants and products if the products have more bonds than the reactants
    assert reac.type == reaction.reactions.Rearrangement
    assert reac.reacs[0].name == 'h3_trigonal'
    assert reac.prods[0].name == 'h3_linear'


def test_reaction_identical_reac_prods():

    h2_reaction = reaction.Reaction(hh_reactant, hh_product)

    h2_reaction.locate_transition_state()
    assert h2_reaction.ts is None


def test_bad_balance():

    with pytest.raises(UnbalancedReaction):
        reaction.Reaction(mol1=h1, mol2=hh_product)

    h_minus = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)], charge=-1)
    with pytest.raises(UnbalancedReaction):
        reaction.Reaction(h1, h_minus, hh_product)

    h1_water = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)], solvent_name='water')
    h2_water = reaction.Reactant(name='h2', atoms=[Atom('H', 1.0, 0.0, 0.0)], solvent_name='water')
    hh_thf = reaction.Product(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)], solvent_name='thf')

    with pytest.raises(SolventsDontMatch):
        reaction.Reaction(h1_water, h2_water, hh_thf)


def test_calc_delta_e():

    r1 = reaction.Reactant(name='h', atoms=[Atom('H', 0.0, 0.0, 0.0)])
    r1.energy = -0.5

    r2 = reaction.Reactant(name='h', atoms=[Atom('H', 0.0, 0.0, 0.0)])
    r2.energy = -0.5

    ts = TS()
    ts.energy = -0.8

    p = reaction.Product(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)])
    p.energy = -1.0

    reac = reaction.Reaction(r1, r2, p)
    reac.solvent_sphere_energy = 0
    reac.ts = ts

    assert -1E-6 < reac.calc_delta_e() < 1E-6
    assert 0.2 - 1E-6 < reac.calc_delta_e_ddagger() < 0.2 + 1E-6
