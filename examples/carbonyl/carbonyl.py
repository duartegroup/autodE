from autode import *

Config.n_cores = 2

acetone = Reactant(name='acetone', smiles='CC(C)=O')
cyanide = Reactant(name='cyanide', smiles='[C-]#N')
cyanopropanolate = Product(name='2-cyanopropan-2-olate', smiles='CC(C)(C#N)[O-]')

reaction = Reaction(acetone, cyanide, cyanopropanolate, name='carbonyl', solvent_name='water')
reaction.calculate_reaction_profile()
