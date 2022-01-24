import autode as ade

ade.Config.n_cores = 8
ade.Config.hcode = 'orca'

if not ade.methods.ORCA().available:
    exit('This example requires an ORCA install')

# Use a basis set with diffuse functions this reaction
ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'

# create a reaction for the addition of CN- to acetone and calculate
rxn = ade.Reaction('CC(C)=O.[C-]#N>>CC([O-])(C#N)C', solvent_name='water')
rxn.calculate_reaction_profile()

print('∆E_r =', rxn.delta('E').to('kcal mol-1'))
print('∆E‡ =', rxn.delta('E‡').to('kcal mol-1'))
print('TS imag freq:', rxn.ts.imaginary_frequencies[0])
