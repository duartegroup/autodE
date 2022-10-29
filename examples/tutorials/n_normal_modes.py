import autode as ade

orca = ade.methods.ORCA()
if not orca.is_available:
    exit("ORCA must be available for this example")

# Optimise and calculate the Hessian for a water molecule
h2o = ade.Molecule(smiles="O")
h2o.optimise(method=orca, keywords="Opt BP86 def2-SV(P)")
h2o.calc_hessian(method=orca, keywords="Freq BP86 def2-SV(P)")

print("Number of total frequencies is:      ", 3 * h2o.n_atoms)
print("Number of vibrational frequencies is:", len(h2o.vib_frequencies))
print(
    "Frequencies in wave-numbers:         ",
    [float(nu) for nu in h2o.vib_frequencies],
)

# Now, generate a set of normal mode-displaced h2o molecules displaced along
# the highest frequency normal mode (index 8), where 0-2 are translations
# 3-5 are rotations and (6, 7, 8) are vibrations
mode = h2o.normal_mode(8)

for i in range(30):
    h2o.coordinates += 0.01 * mode
    h2o.print_xyz_file(filename="h2o_mode2.xyz", append=True)
