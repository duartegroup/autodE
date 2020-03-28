from autode import *

Config.n_cores = 8

methoxide = Reactant(name='methoxide', smiles='C[O-]')
propyl_chloride = Reactant(name='propyl_chloride', smiles='CCCCl')
chloride = Product(name='Cl-', smiles='[Cl-]')
propene = Product(name='propene', smiles='CC=C')
methanol = Product(name='methanol', smiles='CO')

reaction = Reaction(methoxide, propyl_chloride, chloride, propene, methanol,
                    name='elimination', solvent_name='water')
reaction.calculate_reaction_profile()
