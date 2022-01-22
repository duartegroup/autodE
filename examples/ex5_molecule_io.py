import autode as ade

# Molecules can be initialised directly from 3D structures
serine = ade.Molecule('serine.xyz')

# molecules initialised from .xyz files default to neural singlets
print('Name:                    ', serine.name)
print('Charge:                  ', serine.charge)
print('Spin multiplicity:       ', serine.mult)
print('Is solvated?:            ', serine.solvent is not None)

# dihedrals can also be evaluated evaluated
symbols = "-".join(serine.atoms[i].atomic_symbol for i in (0, 1, 2, 3))
print(f'{symbols} dihedral:        ', serine.dihedral(0, 1, 2, 3), 'radians')

# an estimated molecular graph is initialised.
# NOTE: This will be less accurate for organometallic species
print('Bond matrix for the first 4 atoms:\n', serine.bond_matrix[:4, :4])

# molecules also have a has_same_connectivity_as method, which
# checks if the molecular graph is isomorphic to another
blank_mol = ade.Molecule()
print('Num atoms in a empty mol:', blank_mol.n_atoms)
print('Molecular graph is empty:', serine.has_same_connectivity_as(blank_mol))
