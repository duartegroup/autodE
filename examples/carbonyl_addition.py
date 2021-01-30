import autode as ade

ade.Config.n_cores = 8

# For this reaction involving anionic CN- use a basis set with diffuse
# functions
ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'


reaction = ade.Reaction('CC(C)=O.[C-]#N>>CC([O-])(C#N)C', solvent_name='water')
reaction.calculate_reaction_profile()
