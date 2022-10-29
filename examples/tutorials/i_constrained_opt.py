import autode as ade

xtb = ade.methods.XTB()
if not xtb.is_available:
    exit("This example requires an XTB install")

# Constrained optimisations are possible by setting a molecule's constraints
# attribute, for example to calculate the relaxed PES for H-transfer from
# the neutral form of serine to the zwitterion

serine = ade.Molecule("_data/serine.xyz", solvent_name="water")

print("Current N-H distance (Å):", serine.distance(0, 13))

energies = []
for r in (2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0):
    # Set the distance constraint between atoms 0 and 13
    serine.constraints.distance = {(0, 13): r}

    # optimise with XTB
    serine.optimise(method=xtb)

    # and append the energies to a list
    energies.append(serine.energy)

print("Final N-H distance is:  ", serine.distance(0, 13))
print("Energies along the path:", energies)

# Cartesian coordinates can also be fixed. For example, to optimise BH3
# while keeping two H atoms 2 Å apart

gaussian = ade.methods.G09()
if not gaussian.is_available:
    exit("This part requires a Gaussian install")

bh3 = ade.Molecule(
    atoms=[
        ade.Atom("B", y=0.1),
        ade.Atom("H", x=-1.0),
        ade.Atom("H", x=1.0),
        ade.Atom("H", y=1.1),
    ]
)

print("Current H-B-H angle (º):  ", bh3.angle(1, 0, 2).to("º"))

# Set the constraints and do the optimisation
bh3.constraints.cartesian = [1, 2]
bh3.optimise(method=gaussian)

print("Optimised H-B-H angle (º):", bh3.angle(1, 0, 2).to("º"))
