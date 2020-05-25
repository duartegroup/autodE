from autode import *

Config.n_cores = 4

flouride = Reactant(name='F-', smiles='[F-]')
methyl_chloride = Reactant(name='CH3Cl', smiles='ClC')
chloride = Product(name='Cl-', smiles='[Cl-]')
methyl_flouride = Product(name='CH3F', smiles='CF')

reaction = Reaction(flouride, methyl_chloride, chloride, methyl_flouride, name='sn2', solvent_name='water')
reaction.calculate_reaction_profile()
