from autode import *

Config.n_cores = 8

flouride = Reactant(name='F-', smiles='[F-]', solvent_name='water')
methyl_chloride = Reactant(name='CH3Cl', smiles='ClC', solvent_name='water')
chloride = Product(name='Cl-', smiles='[Cl-]', solvent_name='water')
methyl_flouride = Product(name='CH3F', smiles='CF', solvent_name='water')

reaction = Reaction(flouride, methyl_chloride, chloride, methyl_flouride, name='sn2')
reaction.calculate_reaction_profile()
