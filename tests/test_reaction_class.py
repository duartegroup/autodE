from autode import reaction


def test_reaction_class():
    #h + h > h2
    h1 = reaction.Reactant(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    h2 = reaction.Reactant(name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    hh = reaction.Product(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    hh_reac = reaction.Reaction(mol1=h1, mol2=h2, mol3=hh, name='h2_assoc')

    assert hh_reac.type == reaction.reactions.Dissociation
    assert len(hh_reac.prods) == 2
    assert len(hh_reac.reacs) == 1
    assert hh_reac.ts == None
    assert hh_reac.tss == []
    assert hh_reac.name == 'h2_assoc'

    #h + h2 > h2 + h
    hh_reactant = reaction.Reactant(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    h_prod = reaction.Product(name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    h_sub = reaction.Reaction(mol1=h1, mol2=hh_reactant, mol3=hh, mol4=h_prod)

    assert h_sub.type == reaction.reactions.Substitution
    assert h_sub.name == 'reaction'
