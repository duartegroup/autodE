import autode as ade
import matplotlib.pyplot as plt

water = ade.Molecule(name="H2O", smiles="O")
# water.atoms = [[O, x, y, z], [H, x', y', z'], [H, x'', y'', z'']]

# Initialise the unrelaxed potential energy surface over the
# O-H bond from 0.65 Å to 2.0 Å in 20 steps
pes = ade.pes.UnRelaxedPES1D(species=water, rs={(0, 1): (0.65, 2.0, 20)})

# Calculate the surface using the XTB tight-binding DFT method
pes.calculate(method=ade.methods.XTB())

# Finally, plot the surface using matplotlib
plt.plot(pes.r1, pes.relative_energies.to("kcal mol-1"), marker="o")

plt.ylabel("ΔE / kcal mol$^{-1}$")
plt.xlabel("r / Å")
plt.savefig("OH_PES_unrelaxed.png", dpi=400)
