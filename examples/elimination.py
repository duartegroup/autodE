from autode import *

Config.n_cores = 4

fluoride = Reactant(name='F-', xyzs=[['F', 0.0,0.0,0.0]], charge=-1)
propyl_chloride = Reactant(name='ethyl_chloride', smiles='CCCCl')
chloride = Product(name='Cl-', xyzs=[['Cl',0.0,0.0,0.0]], charge=-1)
propene = Product(name='propene', smiles='CC=C')
hydrogen_fluoride = Product(name='HF', smiles='[H]F')

reaction = Reaction(fluoride, propyl_chloride, chloride, propene, hydrogen_fluoride, name='elimination')
reaction.calculate_reaction_profile()
