import autode as ade

ade.Config.n_cores = 8

ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'

methoxide = ade.Reactant(name='methoxide', smiles='C[O-]')
propyl_chloride = ade.Reactant(name='propyl_chloride', smiles='CCCCl')
chloride = ade.Product(name='Cl-', smiles='[Cl-]')
propene = ade.Product(name='propene', smiles='CC=C')
methanol = ade.Product(name='methanol', smiles='CO')

reaction = ade.Reaction(methoxide, propyl_chloride, chloride, propene, methanol,
                        name='E2', solvent_name='water')
reaction.calculate_reaction_profile()
