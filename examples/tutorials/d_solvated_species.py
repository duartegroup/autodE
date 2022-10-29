import autode as ade

# Solvated species can be initialised with
h2 = ade.Species(
    name="h2_in_water",
    charge=0,
    mult=1,
    atoms=[ade.Atom("H"), ade.Atom("H", 0.77)],
    solvent_name="water",
)

print("H2 is solvated in:     ", h2.solvent)

# which are by default implicit solvated
print("Is solvated implicitly:", h2.is_implicitly_solvated)

# the associated solvent has properties, like ε
print("The dielectric is:     ", h2.solvent.dielectric)

# the solvent can be converted to explicit with
h2.explicitly_solvate(num=10)

print("Is solvated explicitly:", h2.is_explicitly_solvated)
print("Number of water atoms: ", h2.solvent.n_atoms)

# the whole solvated system can be printed
h2.print_xyz_file(filename="H2_solv.xyz")
