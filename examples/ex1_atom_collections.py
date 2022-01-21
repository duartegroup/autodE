from autode.atoms import Atom, Atoms


# Atoms in autodE are lists of atom objects
atoms = Atoms([Atom('H', x=-0.4), Atom('H', x=0.4)])

# whcih have a center of mass
print('Center of mass:', atoms.com)

# and moment of inertia properties
print('MOI matrix:\n', atoms.moi)

# vectors between atoms can be calculated
print('H-H vector:   ', atoms.vector(0, 1))
# NOTE: vectors are numpy arrays

# linearity defined
print('H2 is linear: ', atoms.are_linear())

# and also added together
h4_atoms = atoms + atoms.copy()
print('New num atoms:', len(h4_atoms))
