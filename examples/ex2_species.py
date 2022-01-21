import autode as ade

# Species in autodE are atom collections with a defined charge and
# spin multiplicity (mult)
h2o = ade.Species(name='water', charge=0, mult=1,
                  atoms=[ade.Atom('O'),
                         ade.Atom('H', x=-1.),
                         ade.Atom('H', x=0.21, y=-0.97)])

# they have a number of properties, such as mass
print('Mass(H2O):       ', h2o.mass, h2o.mass.units)

# and the chemical formula
print('Formula:         ', h2o.formula)

# as well as radii
print('Approximate radius:', h2o.radius)

# with functions to calculate distances and angles
print('O-H distance (Å):', h2o.distance(0, 1))
print('H-O-H angle (º): ', h2o.angle(1, 0, 2).to('degrees'))

# to save the structure and generate water.xyz in this directory
h2o.print_xyz_file()
