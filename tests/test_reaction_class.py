from autode import reaction
from autode.transition_states.transition_state import TS
from autode.transition_states.ts_guess import TSguess
from autode.solvent.solvents import water_solv
from autode.exceptions import UnbalancedReaction
import pytest


def test_reaction_class():
    # h + h > mol
    h1 = reaction.Reactant(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    h2 = reaction.Reactant(name='mol', xyzs=[['H', 1.0, 0.0, 0.0]])
    hh = reaction.Product(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    hh_reac = reaction.Reaction(mol1=h1, mol2=h2, mol3=hh, name='h2_assoc')
    hh_reac.solvent_sphere_energy = 0

    h1.energy = 2
    h2.energy = 3
    hh.energy = 1

    assert hh_reac.type == reaction.reactions.Dissociation
    assert len(hh_reac.prods) == 2
    assert len(hh_reac.reacs) == 1
    assert hh_reac.ts is None
    assert hh_reac.tss == []
    assert hh_reac.name == 'h2_assoc'
    assert hh_reac.calc_delta_e() == 4

    # h + mol > mol + h
    hh_reactant = reaction.Reactant(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    h_prod = reaction.Product(name='mol', xyzs=[['H', 1.0, 0.0, 0.0]])
    h_sub = reaction.Reaction(mol1=h1, mol2=hh_reactant, mol3=hh, mol4=h_prod, solvent='water')

    assert h_sub.type == reaction.reactions.Substitution
    assert h_sub.name == 'reaction'
    assert h1.solvent_name == water_solv


def test_check_rearrangement():

    # Linear H3 -> Trigonal H3
    lin_h3 = reaction.Reactant(name='h3_linear', xyzs=[['H', -1.76172,        0.79084,       -0.00832],
                                                       ['H', -2.13052,       0.18085,        0.00494],
                                                       ['H', -1.39867,       1.39880,       -0.00676]])

    trig_h3 = reaction.Product(name='h3_trigonal', xyzs=[['H', -1.76172,       0.79084,       -0.00832],
                                                         ['H', -1.65980,       1.15506,       0.61469],
                                                         ['H', -1.39867,        1.39880,       -0.00676]])
    reac = reaction.Reaction(trig_h3, lin_h3)

    # Should switch reactants and products if the products have more bonds than the reactants
    assert reac.reacs[0].name == 'h3_trigonal'
    assert reac.prods[0].name == 'h3_linear'


def test_reaction_identical_reac_prods():
    # H2 -> H2
    hh_r = reaction.Reactant(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    hh_p = reaction.Product(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    h2_reaction = reaction.Reaction(hh_r, hh_p)

    h2_reaction.locate_transition_state()
    assert h2_reaction.ts is None


def test_bad_balance():
    h1 = reaction.Reactant(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    hh = reaction.Product(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    with pytest.raises(UnbalancedReaction):
        _ = reaction.Reaction(mol1=h1, mol2=hh)

    h1 = reaction.Reactant(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]], charge=-1)
    h2 = reaction.Product(name='mol', xyzs=[['H', 1.0, 0.0, 0.0]])
    with pytest.raises(UnbalancedReaction):
        _ = reaction.Reaction(mol1=h1, mol2=h2)

    h1 = reaction.Reactant(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]], solvent='water')
    h2 = reaction.Reactant(name='mol', xyzs=[['H', 1.0, 0.0, 0.0]], solvent='water')
    hh = reaction.Product(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]], solvent='thf')
    with pytest.raises(UnbalancedReaction):
        _ = reaction.Reaction(mol1=h1, mol2=h2, mol3=hh)


def test_calc_delta_e():

    r1 = reaction.Reactant(name='h', xyzs=[['H', 0.0, 0.0, 0.0]])
    r1.energy = -0.5

    r2 = reaction.Reactant(name='h', xyzs=[['H', 0.0, 0.0, 0.0]])
    r2.energy = -0.5

    ts = TS()
    ts.energy = -0.8

    p = reaction.Product(name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    p.energy = -1.0

    reac = reaction.Reaction(r1, r2, p)
    reac.solvent_sphere_energy = 0
    reac.ts = ts

    assert -1E-6 < reac.calc_delta_e() < 1E-6
    assert 0.2 - 1E-6 < reac.calc_delta_e_ddagger() < 0.2 + 1E-6
