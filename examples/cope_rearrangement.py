import autode as ade

ade.Config.n_cores = 8

reactant = ade.Reactant(name='3,4-dimethylhexa-1,5-diene',
                        smiles='C=C[C@H](C)[C@@H](C)C=C')
product = ade.Product(name='octa-2,6-diene', smiles='C/C=C/CC/C=C/C')

reaction = ade.Reaction(reactant, product, name='cope_rearrangement')
reaction.calculate_reaction_profile()
