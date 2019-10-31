from autode import *

Config.n_cores = 4

reac1 = Reactant(name='methyl_sulfate',
                 smiles='O=S(OC)([O-])=O', solvent='water')
reac2 = Reactant(name='OH-', smiles='[OH-]', solvent='water')
prod1 = Product(name='hydrogen_sulfate',
                smiles='O=S(O)([O-])=O', solvent='water')
prod2 = Product(name='methanolate',
                smiles='C[O-]', solvent='water')

reaction = Reaction(reac1, reac2, prod1, prod2, name='sulfate_hydrolysis')
reaction.calculate_reaction_profile()
