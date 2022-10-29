from autode.atoms import Atom, Atoms

# Atoms in autodE are lists of atom objects
atoms = Atoms([Atom("H", x=-0.4), Atom("H", x=0.4)])

# which have a center of mass
print("Center of mass:", atoms.com)

# and moment of inertia properties
print("MOI matrix:", atoms.moi, sep="\n", end="\n\n")

# vectors between atoms can be calculated. For example the vector between
# atoms 0 and 1
print("H-H vector:    ", atoms.vector(0, 1))
# NOTE: vectors are numpy arrays

# to test for linearity of a set of atoms
print("H2 is linear:  ", atoms.are_linear())

# also copied and added together
h4_atoms = atoms + atoms.copy()
print("New num atoms: ", len(h4_atoms))
