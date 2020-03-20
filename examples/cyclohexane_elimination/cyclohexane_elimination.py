from autode import *

Config.n_cores = 8

methoxide = Reactant(name='methoxide', smiles='C[O-]')
chlorocyclohexane = Reactant(name='chlorocyclohexane', smiles='ClC1CCCCC1')
chloride = Product(name='Cl-', smiles='[Cl-]')
cyclohexene = Product(name='cyclohexene', smiles='C1CCC=CC1')
methanol = Product(name='methanol', smiles='CO')

reaction = Reaction(methoxide, chlorocyclohexane, chloride, cyclohexene, methanol,
                    name='cyclohexane_elimination',
                    solvent_name='water')

reaction.calculate_reaction_profile()
