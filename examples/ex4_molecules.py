import autode as ade

# Molecules in autodE are just like species but can
# be initialised from SMILES strings. To generate methane
methane = ade.Molecule(smiles='C')

print(f'Methane has {methane.n_atoms} atoms, so \n'
      f'has a molecular graph with {methane.graph.number_of_nodes()}\n'
      f'and {methane.graph.number_of_edges()} edges (bonds).')

# The whole molecule can be translated
methane.translate([1.0, 0.0, 0.0])
print('Translated carbon position is:', methane.coordinates[0, :])

# and rotated
methane.rotate(axis=[0.0, 0.0, 1.0],
               theta=1.5)
print('Rotated carbon position is:', methane.coordinates[0, :])

# and calculations performed. To optimise the structure with XTB
xtb = ade.methods.XTB()

if xtb.available:
    methane.optimise(method=xtb)
    print('XTB energy is:           ', methane.energy, methane.energy.units)

# along with single points, for example using ORCA
orca = ade.methods.ORCA()

if orca.available:
    methane.single_point(method=ade.methods.ORCA())
    print('ORCA energy is:          ', methane.energy)

# with all energies available
print('All calculated energies: ', methane.energies)
