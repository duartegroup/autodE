from autode import Molecule
from autode.pes.pes_1d import PES1d
from autode.methods import XTB
import numpy as np

# Initialise the electronic structure method (XTB) and water molecule
xtb = XTB()

water = Molecule(name='H2O', smiles='O')

# Initialise a potential energy surface where the reactant and product is
# the same. A product is required to call pes.products_made() to check products
# are generated somewhere on the surface via graph isomorphism
pes = PES1d(reactant=water, product=water,
            rs=np.linspace(0.65, 2.0, num=20),
            r_idxs=(0, 1))

# Calculate the potential energy surface at the XTB level optimising
# all degrees of freedom other than the O-H distance
pes.calculate(name='OH_PES',
              method=xtb,
              keywords=xtb.keywords.low_opt)

# Print the r values and energies
for i in range(pes.n_points):
    print(f'{pes.rs[i, 0]:.3f} {pes.species[i].energy:.4f}')
