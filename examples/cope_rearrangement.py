from autode import *

Config.n_cores = 4

reactant = Reactant(name='3,4-dimethylhexa-1,5-diene', smiles='C=CC(C(C=C)C)C', solvent='water')
product = Product(name='octa-2,6-diene', smiles='CC=CCCC=CC', solvent='water')

reaction = Reaction(reactant, product, name='cope_rearrangement')
reaction.calculate_reaction_profile()