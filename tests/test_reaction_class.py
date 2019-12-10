from autode import reaction
from autode.transition_states.transition_state import TS
from autode.transition_states.ts_guess import TSguess
import pytest
import os


def test_reaction_class():
    #h + h > h2
    h1 = reaction.Reactant(
        name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    h2 = reaction.Reactant(
        name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    hh = reaction.Product(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    hh_reac = reaction.Reaction(
        mol1=h1, mol2=h2, mol3=hh, name='h2_assoc')

    h1.energy = 2
    h2.energy = 3
    hh.energy = 1

    assert hh_reac.type == reaction.reactions.Dissociation
    assert len(hh_reac.prods) == 2
    assert len(hh_reac.reacs) == 1
    assert hh_reac.ts == None
    assert hh_reac.tss == []
    assert hh_reac.name == 'h2_assoc'
    assert hh_reac.calc_delta_e() == 4

    #h + h2 > h2 + h
    hh_reactant = reaction.Reactant(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    h_prod = reaction.Product(name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    h_sub = reaction.Reaction(
        mol1=h1, mol2=hh_reactant, mol3=hh, mol4=h_prod, solvent='water')

    assert h_sub.type == reaction.reactions.Substitution
    assert h_sub.name == 'reaction'
    assert h1.solvent == 'water'


def test_bad_balance():
    h1 = reaction.Reactant(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    hh = reaction.Product(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    with pytest.raises(SystemExit):
        _ = reaction.Reaction(mol1=h1, mol2=hh)

    h1 = reaction.Reactant(
        name='h1', xyzs=[['H', 0.0, 0.0, 0.0]], charge=-1)
    h2 = reaction.Product(
        name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    with pytest.raises(SystemExit):
        _ = reaction.Reaction(mol1=h1, mol2=h2)

    h1 = reaction.Reactant(
        name='h1', xyzs=[['H', 0.0, 0.0, 0.0]], solvent='water')
    h2 = reaction.Reactant(
        name='h2', xyzs=[['H', 1.0, 0.0, 0.0]], solvent='water')
    hh = reaction.Product(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]], solvent='thf')
    with pytest.raises(SystemExit):
        _ = reaction.Reaction(mol1=h1, mol2=h2, mol3=hh)


def test_clear_files():
    h1 = reaction.Reactant(
        name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    h2 = reaction.Reactant(
        name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    hh = reaction.Product(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    hh_reac = reaction.Reaction(
        mol1=h1, mol2=h2, mol3=hh)
    open('aaa.tmp', 'a').close()
    open('xtbrestart', 'a').close()
    hh_reac.clear_tmp_files()
    hh_reac.clear_xtb_files()
    assert not os.path.exists('aaa.tmp')
    assert not os.path.exists('xtbrestart')
