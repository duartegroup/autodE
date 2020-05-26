from autode import *

Config.n_cores = 8

# For hydroxide to be not too reactive in solution requires diffuse functions
Config.ORCA.keywords.low_opt = ['PBE0', 'D3BJ', 'LooseOpt', 'ma-def2-SVP']
Config.ORCA.keywords.hess = ['PBE0', 'D3BJ', 'Freq', 'ma-def2-SVP']
Config.ORCA.keywords.opt = ['PBE0', 'D3BJ', 'LooseOpt', 'ma-def2-SVP']

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
