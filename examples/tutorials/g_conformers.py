import autode as ade


# Conformers of organic molecules initalised from SMILES strings
# in autodE are generated using RDKit. For example,
pentane = ade.Molecule(smiles="CCCCC")

print("Num. initial conformers:    ", pentane.n_conformers)
print("Initial C-C distance (Å):   ", pentane.distance(0, 1))

# To generate a set of conformers
pentane.populate_conformers(n_confs=10)

print("Num. generated conformers:  ", pentane.n_conformers)
# NOTE: the number of generated conformers is usually smaller than
# the number requested, as they are pruned based on similarity
value = ade.Config.rmsd_threshold
print("Default pruning tolerance:  ", value, value.units)

# To find the lowest energy conformer by optimising at XTB then
# re-optimising the unique ones at a higher level
xtb = ade.methods.XTB()
g09 = ade.methods.G09()

if not (xtb.is_available and g09.is_available):
    exit(
        "Cannot run conformer optimisation without both an XTB "
        "and Gaussian09 install"
    )

print(
    f"Generating {ade.Config.num_conformers} conformers "
    f"then pruning based on energy"
)
pentane.find_lowest_energy_conformer(lmethod=xtb, hmethod=g09)

# find_lowest_energy_conformer will set the molecule's geometry and energy
print("Optimised C-C distance (Å): ", pentane.distance(0, 1))
print("Potential energy:           ", pentane.energy, pentane.energy.units)
print("Pruned number of conformers:", pentane.n_conformers)
