from autode import *

Config.n_cores = 4

flouride = Reactant(
    name='F-', xyzs=[['F', 0.0, 0.0, 0.0]], solvent='water', charge=-1)
methyl_chloride = Reactant(
    name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent='water')
chloride = Product(
    name='Cl-', xyzs=[['Cl', 0.0, 0.0, 0.0]], solvent='water', charge=-1)
methyl_flouride = Product(
    name='CH3F', smiles='[H]C([H])(F)[H]', solvent='water')

reaction = Reaction(flouride, methyl_chloride, chloride,
                    methyl_flouride, name='sn2')
reaction.calculate_reaction_profile()
