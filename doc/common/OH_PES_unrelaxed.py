import autode as ade
import matplotlib.pyplot as plt
import numpy as np

# Initialise the electronic structure method (XTB)
xtb = ade.methods.XTB()

water = ade.Molecule(name='H2O', smiles='O')
# water.atoms = [[O, x, y, z], [H, x', y', z'], [H, x'', y'', z'']]

# Array of distances (Å) to calculate the energy for
rs = np.linspace(0.65, 2.0, num=20)

# List of energies to be populated
energies = []

for r in rs:

    o_atom, h_atom = water.atoms[:2]
    curr_r = water.distance(0, 1)       # current O-H distance

    # Shift the hydrogen atom to the required distance
    # vector =  (h_atom.coord - o_atom.coord) / curr_r * (r - curr_r)
    vector = (h_atom.coord - o_atom.coord) * (r/curr_r - 1)
    h_atom.translate(vector)

    # Set up and run the calculation
    calc = ade.Calculation(name=f'H2O_scan_{r:.2f}',
                           molecule=water,
                           method=xtb,
                           keywords=xtb.keywords.sp)
    calc.run()

    # Get the potential energy from the calculation
    energy = calc.get_energy()
    energies.append(energy)

# Plot the relative energy against the distance. 627.5 kcal mol-1 Ha-1
min_e_kcal = min(energies).to('kcal mol-1')
rel_energies = [e.to('kcal mol-1') - min_e_kcal for e in energies]

plt.plot(rs, rel_energies, marker='o')
plt.ylabel('ΔE / kcal mol$^{-1}$')
plt.xlabel('r / Å')
plt.tight_layout()
plt.savefig('OH_PES_unrelaxed.pdf')
