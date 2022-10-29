import autode as ade
from autode.conformers import conf_gen, Conformer

# Initialise the complex from a .xyz file containing a square planar structure
vaskas = ade.Molecule("vaskas.xyz")

# Set up some distance constraints where the keys are the atom indexes and
# the value the distance in Å. Fixing the Cl-P, Cl-P and Cl-C(=O) distances
# enforces a square planar geometry
distance_constraints = {
    (1, 2): vaskas.distance(1, 2),
    (1, 3): vaskas.distance(1, 3),
    (1, 4): vaskas.distance(1, 4),
}

# Generate 5 conformers
for n in range(5):

    # Apply random displacements to each atom and minimise under a bonded +
    # repulsive forcefield including the distance constraints
    atoms = conf_gen.get_simanl_atoms(
        species=vaskas, dist_consts=distance_constraints, conf_n=n
    )

    # Generate a conformer from these atoms then optimise with XTB
    conformer = Conformer(name=f"vaskas_conf{n}", atoms=atoms)

    conformer.optimise(method=ade.methods.XTB())
    conformer.print_xyz_file()
