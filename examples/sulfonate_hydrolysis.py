from autode import *

Config.n_cores = 4

reac1 = Reactant(name='methyl_methanesulfonate',
                 smiles='O=S(OC)(C)=O', solvent='water')
reac2 = Reactant(name='OH-', smiles='[OH-]', solvent='water')
prod1 = Product(name='methanesulfonic_acid',
                smiles='O=S(O)(C)=O', solvent='water')
prod2 = Product(name='methoxide', smiles='C[O-]', solvent='water')

reaction = Reaction(reac1, reac2, prod1, prod2, name='sulfonate_hydrolysis')
reaction.calculate_reaction_profile()
