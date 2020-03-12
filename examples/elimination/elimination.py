from autode import *

Config.n_cores = 4

methoxide = Reactant(name='methoxide', smiles='C[O-]', solvent_name='water')
propyl_chloride = Reactant(name='propyl_chloride', smiles='CCCCl', solvent_name='water')
chloride = Product(name='Cl-', smiles='[Cl-]', solvent_name='water')
propene = Product(name='propene', smiles='CC=C', solvent_name='water')
methanol = Product(name='methanol', smiles='CO', solvent_name='water')

reaction = Reaction(methoxide, propyl_chloride, chloride, propene, methanol, name='elimination')
reaction.calculate_reaction_profile()
