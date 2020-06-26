from autode import Molecule
from autode.calculation import Calculation
from autode.methods import XTB
import matplotlib.pyplot as plt
import numpy as np

# Initialise the electronic structure method (XTB)
xtb = XTB()

water = Molecule(name='H2O', smiles='O')
# water.atoms = [[O, x, y, z], [H, x', y', z'], [H, x'', y'', z'']]

# Array of distances (Å) to calculate the energy for
rs = np.linspace(0.65, 2.0, num=20)

# List of energies to be populated
energies = []

for r in rs:

    o_atom, h_atom = water.atoms[:2]
    curr_r = water.get_distance(0, 1)       # current O-H distance

    # Shift the hydrogen atom to the required distance
    # vector =  (h_atom.coord - o_atom.coord) / curr_r * (r - curr_r)
    vector = (h_atom.coord - o_atom.coord) * (r/curr_r - 1)
    h_atom.translate(vector)

    # Set up and run the calculation
    calc = Calculation(name=f'H2O_scan_{r:.2f}',
                       molecule=water,
                       method=xtb,
                       keywords=xtb.keywords.sp)
    calc.run()

    # Get the potential energy from the calculation
    energy = calc.get_energy()
    energies.append(energy)

# Plot the relative energy against the distance. 627.5 kcal mol-1 Ha-1
rel_energies = 627.5 * np.array([e - min(energies) for e in energies])

plt.plot(rs, rel_energies, marker='o')
plt.ylabel('ΔE / kcal mol$^{-1}$')
plt.xlabel('r / Å')
plt.savefig('OH_PES_unrelaxed.png')
