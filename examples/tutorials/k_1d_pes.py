import autode as ade

xtb = ade.methods.XTB()
if not xtb.is_available:
    exit("This example requires an XTB install")

# One dimensional potential energy surfaces without optimising other
# coordinates can be calculated. For example, for the C-C stretch in ethane

ethane = ade.Molecule(smiles="CC")
ethane.optimise(method=xtb)

# Determine the initial C-C bond length for the scan
atom_idxs = (0, 1)
r = ethane.distance(*atom_idxs) - 0.05

# Set up the PES scan over the 0-1 distance (C-C) from r to 2.0 Å in 20 steps
pes = ade.pes.UnRelaxedPES1D(ethane, rs={atom_idxs: (r, 2.0, 20)})
pes.calculate(method=xtb)

print("Energies:", pes.relative_energies.to("kcal mol-1"))

# PESs can also be plotted. To save PES.pdf in the current directory
pes.plot()
