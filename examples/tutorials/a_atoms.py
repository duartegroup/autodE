import autode as ade

# Create a carbon atom
atom = ade.Atom("C")

# autodE atoms have a position
print("Position:  ", atom.coord)

# and useful properties, like the atomic number. See all of these properties
# here: https://duartegroup.github.io/autodE/reference/atoms.html
print("Z:         ", atom.atomic_number)

# and can be translated by a vector, for example 1 Å along the x axis
atom.translate([1.0, 0.0, 0.0])
print("Position:  ", atom.coord)

# or rotated
atom.rotate(axis=[0.0, 0.0, 1.0], theta=3.1415)  # z axis  # π radians
print("Position:  ", atom.coord)

# by default distances are in angstroms
atom.coord = [0.5, 0.0, 0.0]
print("Units are: ", atom.coord.units)

# and can be converted to others. For example, Bohr
coord_in_a0 = atom.coord.to("bohr")
print("Coordinate:", coord_in_a0, coord_in_a0.units)

# atoms can also be initialised at different positions
atom = ade.Atom("H", x=1.0, y=2.0, z=3.0)

# and their representation (repr) printed
print("H atom:    ", repr(atom))
