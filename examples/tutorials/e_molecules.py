import autode as ade

# Molecules in autodE are just like species but can
# be initialised from SMILES strings. To generate methane
methane = ade.Molecule(smiles="C")

print(
    f"Methane has {methane.n_atoms} atoms, so \n"
    f"has a molecular graph with {methane.graph.number_of_nodes()}\n"
    f"nodes and {methane.graph.number_of_edges()} edges (bonds)."
)

# The whole molecule can be translated
methane.translate([1.0, 0.0, 0.0])
print("Translated carbon position is:", methane.coordinates[0, :])
# where the coordinates property is an Nx3 numpy array

# and rotated
methane.rotate(axis=[0.0, 0.0, 1.0], theta=1.5)  # z axis  # radians
print("Rotated carbon position is:   ", methane.coordinates[0, :])

# and calculations performed. To optimise the structure with XTB
xtb = ade.methods.XTB()
print(f"Using {ade.Config.n_cores} cores for an XTB calculation")

if xtb.is_available:
    methane.optimise(method=xtb)
    print("XTB energy (Ha):              ", methane.energy)

# along with single points. For example, using ORCA
orca = ade.methods.ORCA()
print(f"Using {ade.Config.n_cores} cores for an ORCA calculation")

if orca.is_available:
    print(f"Calculating at the: [{orca.keywords.sp}] level of theory")
    methane.single_point(method=orca)
    print("ORCA energy (Ha):             ", methane.energy)

# with all energies available
print("All calculated energies:      ", methane.energies)
