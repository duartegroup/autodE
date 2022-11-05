import autode as ade

ade.Config.lcode = "xtb"
ade.Config.hcode = "g09"

if not (ade.methods.G09().is_available and ade.methods.XTB().is_available):
    exit("This example requires a Gaussian09 and XTB install")

# Full reaction profiles can be calculated by again forming a reaction
# and calling calculate_reaction_profile. Conformers will be searched,
# a TS found and single point energies evaluated. The reaction is defined a
# as a single string, with reactants and products separated by '>>'
rxn = ade.Reaction("CCl.[F-]>>CF.[Cl-]", solvent_name="water")

print(f"Calculating the reaction profile for {rxn.reacs}->{rxn.prods}")
rxn.calculate_reaction_profile()

print("∆E‡ =", rxn.delta("E‡").to("kcal mol-1"))
