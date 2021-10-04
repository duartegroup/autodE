import autode as ade
import matplotlib.pyplot as plt
import numpy as np

# Initialise the electronic structure method (XTB)
xtb = ade.methods.XTB()

water = ade.Molecule(name='H2O', smiles='O')
rs = np.linspace(0.65, 2.0, num=20)

# List of energies to be populated for the single point (unrelaxed)
# and constrained optimisations (relaxed) calculations
sp_energies, opt_energies = [], []

for r in rs:

    o_atom, h_atom = water.atoms[:2]
    curr_r = water.distance(0, 1)       # current O-H distance

    # Shift the hydrogen atom to the required distance
    # vector =  (h_atom.coord - o_atom.coord) / curr_r * (r - curr_r)
    vector = (h_atom.coord - o_atom.coord) * (r/curr_r - 1)
    h_atom.translate(vector)

    # Set up and run the single point energy evaluation
    sp = ade.Calculation(name=f'H2O_scan_unrelaxed_{r:.2f}',
                         molecule=water,
                         method=xtb,
                         keywords=xtb.keywords.sp)
    sp.run()
    sp_energies.append(sp.get_energy())

    # Set up the constrained optimisation calculation where the distance
    # constraints are given as a dictionary keyed with a tuple of atom indexes
    # with the distance as the value
    opt = ade.Calculation(name=f'H2O_scan_relaxed_{r:.2f}',
                          molecule=water,
                          method=xtb,
                          keywords=xtb.keywords.low_opt,
                          distance_constraints={(0, 1): r})
    opt.run()
    opt_energies.append(opt.get_energy())

# Plot the relative energy against the distance. 627.5 kcal mol-1 Ha-1
plt.plot(rs,
         [(e - min(sp_energies)).to('kcal mol-1') for e in sp_energies],
         marker='o',
         label='unrelaxed',
         c='r')

# And the same for the relaxed points
plt.plot(rs,
         [(e - min(opt_energies)).to('kcal mol-1') for e in opt_energies],
         marker='o',
         label='relaxed',
         c='b')

plt.ylabel('ΔE / kcal mol$^{-1}$')
plt.xlabel('r / Å')
plt.legend()
plt.savefig('OH_PES_comparison.pdf')
