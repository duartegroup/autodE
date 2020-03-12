from autode import *

Config.n_cores = 4

methoxide = Reactant(name='methoxide', smiles='C[O-]', solvent_name='water')
chlorocyclohexane = Reactant(name='chlorocyclohexane', smiles='ClC1CCCCC1', solvent_name='water')
chloride = Product(name='Cl-', smiles='[Cl-]', solvent_name='water')
cyclohexene = Product(name='cyclohexene', smiles='C1CCC=CC1', solvent_name='water')
methanol = Product(name='methanol', smiles='CO', solvent_name='water')

reaction = Reaction(methoxide, chlorocyclohexane, chloride, cyclohexene, methanol, name='cyclohexane_elimination')
reaction.calculate_reaction_profile()
