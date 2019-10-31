from autode import *

Config.n_cores = 4

reac1 = Reactant(name='big_reac',
                 smiles='ClC(CC(CC)C(C)C)CC(CC)CC', solvent='water')
reac2 = Reactant(name='OH-', smiles='[OH-]', solvent='water')
prod1 = Product(name='big_prod',
                smiles='OC(CC(CC)C(C)C)CC(CC)CC', solvent='water')
prod2 = Product(name='Cl-',
                smiles='[Cl-]', solvent='water')

reaction = Reaction(reac1, reac2, prod1, prod2, name='big_sn2')
reaction.calculate_reaction_profile()
