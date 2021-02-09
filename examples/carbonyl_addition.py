import autode as ade

ade.Config.n_cores = 8

# Use a basis set with diffuse functions this reaction involving anionic CN-
ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'

# Define a reaction as a single string, with reactants and products seperated
# by '>>'
reaction = ade.Reaction('CC(C)=O.[C-]#N>>CC([O-])(C#N)C', solvent_name='water')
reaction.calculate_reaction_profile()
