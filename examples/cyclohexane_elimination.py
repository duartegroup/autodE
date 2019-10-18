from autode import *

Config.n_cores = 4

methoxide = Reactant(name='methoxide', smiles='C[O-]', solvent='water')
chlorocyclohexane = Reactant(name='chlorocyclohexane', smiles='ClC1CCCCC1', solvent='water')
chloride = Product(name='Cl-', xyzs=[['Cl',0.0,0.0,0.0]], charge=-1, solvent='water')
cyclohexene = Product(name='cyclohexene', smiles='C1CCC=CC1', solvent='water')
methanol = Product(name='methanol', smiles='CO', solvent='water')

reaction = Reaction(methoxide, chlorocyclohexane, chloride, cyclohexene, methanol, name='cyclohexane_elimination')
reaction.calculate_reaction_profile()
