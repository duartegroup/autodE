import autode as ade

# Species in autodE are atom collections with a name,
# defined charge and spin multiplicity (mult). For example, to
# generate water from its three constituent atoms
h2o = ade.Species(
    name="water",
    charge=0,
    mult=1,
    atoms=[
        ade.Atom("O"),
        ade.Atom("H", x=-1.0),
        ade.Atom("H", x=0.21, y=-0.97),
    ],
)

# they have a number of properties, such as mass
print("Mass(H2O):       ", h2o.mass, h2o.mass.units)

# and the chemical formula
print("Formula:         ", h2o.formula)

# as well as radii *not including any van der Walls volume*
print("Approximate radius:", round(h2o.radius, 1), h2o.radius.units)

# with functions to calculate distances and angles between atoms
# For example, the distance between atoms 0 and 1:
print("O-H distance (Å):", h2o.distance(0, 1))
print("H-O-H angle (º): ", h2o.angle(1, 0, 2).to("degrees"))

# to save the structure and generate water.xyz in this directory
h2o.print_xyz_file()
