from autode import Reactant, Product, Reaction, Config
Config.n_cores = 4

r1 = Reactant(smiles='CC(C)=O', name='acetone')
r2 = Reactant(smiles='[C-]#N', name='cn-')
p = Product(smiles='CC([O-])(C#N)C', name='prod')

reaction = Reaction(r1, r2, p, solvent_name='water')
reaction.calculate_reaction_profile()
