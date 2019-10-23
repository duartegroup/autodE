from autode import *

Config.n_cores = 4

acetone = Reactant(name='acetone', smiles='CC(C)=O', solvent='water')
cyanide = Reactant(name='cyanide', smiles='[C-]#N', solvent='water')
cyanopropanolate = Product(
    name='2-cyanopropan-2-olate', smiles='CC(C)(C#N)[O-]', solvent='water')

reaction = Reaction(acetone, cyanide, cyanopropanolate, name='carbonyl')
reaction.calculate_reaction_profile()
