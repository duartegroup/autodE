import autode as ade

ade.Config.n_cores = 10  # Distribute over 10 cores

# PES from the current C-C distances (~1.5 Å) to broken (3.0 Å) in 10 steps
pes = ade.pes.RelaxedPESnD(
    species=ade.Molecule("cyclohexene.xyz"),
    rs={(0, 5): (3.0, 10), (3, 4): (3.0, 10)},
)

pes.calculate(method=ade.methods.XTB())
pes.plot("DA_surface.png")
pes.save("DA_surface.npz")
