import autode as ade
import numpy as np
ade.Config.n_cores = 4

reac = ade.Reactant('DA_r.xyz')
prod = ade.Product('DA_p.xyz')

pes = ade.pes.PES2d(reac, prod,
                    r1s=np.linspace(1.45, 3.0, 10),
                    r1_idxs=(0, 5),
                    r2s=np.linspace(1.45, 3.0, 10),
                    r2_idxs=(3, 4))

pes.calculate(name='da_surface',
              method=ade.methods.XTB(),
              keywords=ade.OptKeywords())

# Save the matrix of energies (Ha) over the grid
np.savetxt('da_surface.txt', pes.energies())
# and print the 2D surface
pes.print_plot()
