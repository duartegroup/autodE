import autode as ade

water = ade.Molecule(name="H2O", smiles="O")

# Initialise a relaxed potential energy surface for the water O-H stretch
# from 0.65 -> 2.0 Å in 15 steps
pes = ade.pes.RelaxedPESnD(species=water, rs={(0, 1): (0.65, 2.0, 15)})

pes.calculate(method=ade.methods.XTB())
pes.plot("OH_PES_relaxed.png")

# PESs can also be saved as compressed numpy objects and reloaded
pes.save("pes.npz")

# For example, reload the PES and print the distances and energies
pes = ade.pes.RelaxedPESnD.from_file("pes.npz")

print("r (Å)   E (Ha)")
for i in range(15):
    print(f"{pes.r1[i]:.4f}", f"{pes[i]:.5f}")
