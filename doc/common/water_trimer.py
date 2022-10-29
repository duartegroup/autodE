import autode as ade

xtb = ade.methods.XTB()

# Number of points on the surface of the sphere for each component
ade.Config.num_complex_sphere_points = 5  # N_s

# and the number of rotations to perform per point on the sphere
ade.Config.num_complex_random_rotations = 3  # N_r

# Total number of conformers ~(N_s × N_r)^(N-1) for N molecules => ~225

# Make a water molecule and optimise at the XTB level
water = ade.Molecule(name="water", smiles="O")
water.optimise(method=xtb)

# Make the NCI complex and find the lowest energy structure
trimer = ade.NCIComplex(water, water, water, name="water_trimer")
trimer.find_lowest_energy_conformer(lmethod=xtb)
trimer.print_xyz_file()
