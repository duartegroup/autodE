from autode import *

Config.n_cores = 4

methoxide = Reactant(name='methoxide', smiles='C[O-]', solvent='water')
propyl_chloride = Reactant(name='ethyl_chloride', smiles='CCCCl', solvent='water')
chloride = Product(name='Cl-', xyzs=[['Cl',0.0,0.0,0.0]], charge=-1, solvent='water')
propene = Product(name='propene', smiles='CC=C', solvent='water')
methanol = Product(name='methanol', smiles='CO', solvent='water')

reaction = Reaction(methoxide, propyl_chloride, chloride, propene, methanol, name='elimination')
reaction.calculate_reaction_profile()
