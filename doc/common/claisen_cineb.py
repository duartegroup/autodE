import autode as ade

reac = ade.Reactant("claisen_r.xyz")
prod = ade.Product("claisen_p.xyz")

# Create an 8 image nudged elastic band with intermediate images interpolated
# from the final points, thus they must be structurally similar
neb = ade.CINEB.from_end_points(reac, prod, num=8)
# minimise with XTB
neb.calculate(method=ade.methods.XTB(), n_cores=4)

# print the geometry of the peak species
print("Found a peak: ", neb.images.contains_peak)
neb.peak_species.print_xyz_file(filename="peak.xyz")
