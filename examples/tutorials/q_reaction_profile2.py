import autode as ade

ade.Config.n_cores = 8
ade.Config.hcode = "orca"

if not ade.methods.ORCA().is_available:
    exit("This example requires an ORCA install")

# Use a basis set with diffuse functions this reaction
ade.Config.ORCA.keywords.set_opt_basis_set("ma-def2-SVP")
ade.Config.ORCA.keywords.sp.basis_set = "ma-def2-TZVP"

# create a reaction for the addition of CN- to acetone and calculate
rxn = ade.Reaction("CC(C)=O.[C-]#N>>CC([O-])(C#N)C", solvent_name="water")

# Calculating a reaction profile is also possible including the energies
# of the pre-reaction association complexes (with_complexes=True), including
# the free energy (∆G, ∆G‡) using qRRHO entropies (free_energy=True) or
# reaction and activation enthalpies (∆H, ∆H‡) using enthalpy=True
rxn.calculate_reaction_profile(
    # with_complexes=False,
    # free_energy=False,
    # enthalpy=False
)

print("∆E_r (kcal mol-1) = ", rxn.delta("E").to("kcal mol-1"))
print("∆E‡ (kcal mol-1)  = ", rxn.delta("E‡").to("kcal mol-1"))
print("TS imaginary freq = ", rxn.ts.imaginary_frequencies[0])
