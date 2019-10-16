from autode import *

Config.n_cores = 4

dimethylbut_two_ene = Reactant(name='2,3-dimethylbut-2-ene', smiles='CC(C)=C(C)C', solvent='water')
diazene = Reactant(name='diazene', smiles='N=N', solvent='water')
dimethylbutane = Product(name='2,3-dimethylbutane', smiles='CC(C(C)C)C', solvent='water')
nitrogen = Product(name='N2', smiles='N#N', solvent='water')

reaction = Reaction(dimethylbut_two_ene, diazene, dimethylbutane, nitrogen, name='diazene_addition')
reaction.calculate_reaction_profile()