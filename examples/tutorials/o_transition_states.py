import autode as ade

# Use ORCA DFT optimisations
ade.Config.lcode = "xtb"
ade.Config.hcode = "orca"

if not (ade.methods.ORCA().is_available and ade.methods.XTB().is_available):
    exit("This example requires an ORCA and XTB install")

# Use 8 cores for the calculations
ade.Config.n_cores = 8

# Locating transition states (TSs) in autodE requires defining a reaction.
# For example, the TS for a key step in a Beckmann rearrangement can be
# calculated with
r1 = ade.Reactant("_data/Beckmann/reactant.xyz", charge=1)
p1 = ade.Product("_data/Beckmann/product.xyz", charge=1)
p2 = ade.Product("_data/Beckmann/water.xyz")

# Form the reaction and locate the transition state
rxn = ade.Reaction(r1, p1, p2)

print("Locating the TS for a Beckmann rearrangement...")
rxn.locate_transition_state()

if rxn.ts is not None:
    print("TS has been found!")
    print("Imaginary frequency: ", rxn.ts.imaginary_frequencies[0])
    rxn.ts.print_xyz_file(filename="TS_beckmann.xyz")

print("Total number of found TSs:", len(rxn.tss))
for ts in rxn.tss:
    print(f"E(TS {ts.name}) = ", ts.energy)
