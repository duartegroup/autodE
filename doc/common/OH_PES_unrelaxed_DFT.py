from autode import Molecule, SinglePointKeywords
from autode.calculation import Calculation
from autode.methods import ORCA
import matplotlib.pyplot as plt
import numpy as np

# Initialise the electronic structure method and a list of different
# single point energy keywords
orca = ORCA()

keywords_list = {'PBE': SinglePointKeywords(['PBE', 'def2-SVP']),
                 'PBE0': SinglePointKeywords(['PBE0', 'def2-SVP']),
                 'B3LYP': SinglePointKeywords(['B3LYP', 'def2-SVP'])}

water = Molecule(name='H2O', smiles='O')

# For the three different DFT functionals calculate the PES and plot the line
for dft_name, keywords in keywords_list.items():

    # Create arrays for OH distances and their energies
    rs = np.linspace(0.65, 2.0, num=15)
    energies = []

    # Calculate the energy array
    for r in rs:

        o_atom, h_atom = water.atoms[:2]
        curr_r = water.get_distance(0, 1)

        vector = (h_atom.coord - o_atom.coord) * (r/curr_r - 1)
        h_atom.translate(vector)

        # Set up and run the calculation
        calc = Calculation(name=f'H2O_scan_{r:.2f}',
                           molecule=water,
                           method=orca,
                           keywords=keywords)
        calc.run()

        # Get the potential energy from the calculation
        energy = calc.get_energy()
        energies.append(energy)

    # Plot the relative energy against the distance. 627.5 kcal mol-1 Ha-1
    rel_energies = 627.5 * np.array([e - min(energies) for e in energies])
    plt.plot(rs, rel_energies, marker='o', label=dft_name)

# Add labels to the plot and save the figure
plt.ylabel('ΔE / kcal mol$^{-1}$')
plt.xlabel('r / Å')
plt.legend()
plt.savefig('OH_PES_unrelaxed2.png')
