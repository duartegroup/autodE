import autode as ade

ade.Config.n_cores = 8

flouride = ade.Reactant(name='F-', smiles='[F-]')
methyl_chloride = ade.Reactant(name='CH3Cl', smiles='ClC')
chloride = ade.Product(name='Cl-', smiles='[Cl-]')
methyl_flouride = ade.Product(name='CH3F', smiles='CF')

reaction = ade.Reaction(flouride, methyl_chloride, chloride, methyl_flouride,
                        name='sn2', solvent_name='water')
reaction.calculate_reaction_profile()
