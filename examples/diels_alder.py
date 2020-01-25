from autode import *

Config.n_cores = 4

butadiene = Reactant(name='butadiene', smiles='C=CC=C')
ethene = Reactant(name='ethene', smiles='C=C')
cyclohexene = Product(name='cyclohexene', smiles='C1C=CCCC1')

reaction = Reaction(butadiene, ethene, cyclohexene, name='diels_alder')
reaction.calculate_reaction_profile()
