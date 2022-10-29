import autode as ade

pes = ade.pes.RelaxedPESnD.from_file("DA_surface.npz")
pes.plot("DA_surface_interpolated.png", interp_factor=4)
