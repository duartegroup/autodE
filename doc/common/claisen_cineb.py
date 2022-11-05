import autode as ade

reac = ade.Reactant("claisen_r.xyz")
prod = ade.Product("claisen_p.xyz")

# Create an 8 image nudged elastic band with intermediate images interpolated
# from the final points, thus they must be structurally similar
neb = ade.neb.CINEB(initial_species=reac, final_species=prod, num=8)
# minimise with XTB
neb.calculate(method=ade.methods.XTB(), n_cores=4)

# print the geometry of the peak species
peak_species = neb.get_species_saddle_point()
peak_species.print_xyz_file(filename="peak.xyz")
