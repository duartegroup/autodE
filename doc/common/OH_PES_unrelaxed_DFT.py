import autode as ade
import matplotlib.pyplot as plt
import numpy as np

# Initialise the electronic structure method and a list of different
# single point energy keywords
orca = ade.methods.ORCA()

keywords_list = {'PBE': ade.SinglePointKeywords(['PBE', 'def2-SVP']),
                 'PBE0': ade.SinglePointKeywords(['PBE0', 'def2-SVP']),
                 'B3LYP': ade.SinglePointKeywords(['B3LYP', 'def2-SVP'])}

water = ade.Molecule(name='H2O', smiles='O')

# For the three different DFT functionals calculate the PES and plot the line
for dft_name, keywords in keywords_list.items():

    # Create arrays for OH distances and their energies
    rs = np.linspace(0.65, 2.0, num=15)
    energies = []

    # Calculate the energy array
    for r in rs:

        o_atom, h_atom = water.atoms[:2]
        curr_r = water.distance(0, 1)

        vector = (h_atom.coord - o_atom.coord) * (r/curr_r - 1)
        h_atom.translate(vector)

        # Set up and run the calculation
        calc = ade.Calculation(name=f'H2O_scan_{r:.2f}',
                               molecule=water,
                               method=orca,
                               keywords=keywords)
        calc.run()

        # Get the potential energy from the calculation
        energy = calc.get_energy()
        energies.append(energy)

    # Plot the relative energy against the distance. 627.5 kcal mol-1 Ha-1
    min_e_kcal = min(energies).to('kcal mol-1')
    plt.plot(rs,
             [e.to('kcal mol-1') - min_e_kcal for e in energies],
             marker='o',
             label=dft_name)

# Add labels to the plot and save the figure
plt.ylabel('ΔE / kcal mol$^{-1}$')
plt.xlabel('r / Å')
plt.legend()
plt.tight_layout()
plt.savefig('OH_PES_unrelaxed2.pdf')
