import autode as ade
import matplotlib.pyplot as plt

# Initialise the PES over the O-H bond 0.65 -> 2.0 Å
pes = ade.pes.UnRelaxedPES1D(
    species=ade.Molecule(name="H2O", smiles="O"), rs={(0, 1): (0.65, 2.0, 20)}
)

# For the three different DFT functionals calculate the PES and plot the line
for functional in ("PBE", "PBE0", "B3LYP"):

    pes.calculate(method=ade.methods.ORCA(), keywords=[functional, "def2-SVP"])

    plt.plot(
        pes.r1,
        pes.relative_energies.to("kcal mol-1"),
        marker="o",
        label=functional,
    )

# Add labels to the plot and save the figure
plt.ylabel("ΔE / kcal mol$^{-1}$")
plt.xlabel("r / Å")
plt.legend()
plt.tight_layout()
plt.savefig("OH_PES_unrelaxed_DFT.png", dpi=400)
