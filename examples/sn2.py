import autode as ade

ade.Config.n_cores = 8

fluoride = ade.Reactant(name='F-', smiles='[F-]')
methyl_chloride = ade.Reactant(name='CH3Cl', smiles='ClC')
chloride = ade.Product(name='Cl-', smiles='[Cl-]')
methyl_fluoride = ade.Product(name='CH3F', smiles='CF')

reaction = ade.Reaction(fluoride, methyl_chloride, chloride, methyl_fluoride,
                        name='sn2', solvent_name='water')
reaction.calculate_reaction_profile()
