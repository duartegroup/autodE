from autode import *

Config.n_cores = 4

flouride = Reactant(name='F-', smiles='[F-]', solvent='water')
methyl_chloride = Reactant(name='CH3Cl', smiles='ClC', solvent='water')
chloride = Product(name='Cl-', smiles='[Cl-]', solvent='water')
methyl_flouride = Product(name='CH3F', smiles='CF', solvent='water')

reaction = Reaction(flouride, methyl_chloride, chloride,methyl_flouride, name='sn2')
reaction.calculate_reaction_profile()
