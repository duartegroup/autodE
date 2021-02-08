import autode as ade

ade.Config.n_cores = 8
ade.Config.hcode = 'g09'       # Use Gaussian09 as the high-level method

# For hydroxide to be not too reactive requires diffuse functions; set all the
ade.Config.G09.keywords.set_opt_basis_set('6-31+G(d)')
ade.Config.G09.keywords.sp.basis_set = '6-311+G(d,p)'

# Set up the first step in the hydrolysis of the ester, attack of OH- to get a
# tetrahedral intermediate
r1 = ade.Reactant(name='ester', smiles='CC(OC)=O')
r2 = ade.Reactant(name='hydroxide', smiles='[OH-]')
tet_int = ade.Product(name='tet_intermediate', smiles='CC([O-])(OC)O')

step1 = ade.Reaction(r1, r2, tet_int, solvent_name='water')

# Second step is collapse of the tetrahedral intermediate to the acid and
# methoxide
tet_int = ade.Reactant(name='tet_intermediate', smiles='CC([O-])(OC)O')
p1 = ade.Product(name='acid', smiles='CC(O)=O')
p2 = ade.Product(name='methodixe', smiles='[O-]C')

step2 = ade.Reaction(tet_int, p1, p2, solvent_name='water')

# Calculate the reactions in sequence, so the conformers of the tetrahedral
# intermediate do not need to be found again
reaction = ade.MultiStepReaction(step1, step2)
reaction.calculate_reaction_profile()
