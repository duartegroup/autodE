from autode import *

Config.n_cores = 8

# For hydroxide to be not too reactive in solution requires diffuse functions;
# set all the optimisation and the hessian keywords to include the ma-def2-SVP
# basis set, assuming the hmethod is ORCA..
Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
Config.ORCA.keywords.sp.basis_set = 'ma-def2-SVP'

# Set up the first step in the hydrolysis of the ester, attack of OH- to get a
# tetrahedral intermediate
r1 = Reactant(name='ester', smiles='CC(OC)=O')
r2 = Reactant(name='hydroxide', smiles='[OH-]')
tet_int = Product(name='tet_intermediate', smiles='CC([O-])(OC)O')

step1 = Reaction(r1, r2, tet_int, solvent_name='water')

# Second step is collapse of the tetrahedral intermediate to the acid and
# methoxide
tet_int = Reactant(name='tet_intermediate', smiles='CC([O-])(OC)O')
p1 = Product(name='acid', smiles='CC(O)=O')
p2 = Product(name='methodixe', smiles='[O-]C')

step2 = Reaction(tet_int, p1, p2, solvent_name='water')

# Calculate the reactions in sequence, so the conformers of the tetrahedral
# intermediate do not need to be found again
reaction = MultiStepReaction(step1, step2)
reaction.calculate_reaction_profile()
