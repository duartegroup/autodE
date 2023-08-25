import sys
import autode as ade
ade.Config.n_cores = 4
ade.Config.max_core = 2000

# angle constraints
water = ade.Molecule(name='water', smiles='O')

water.constraints.angular = {(0,1,2): 104.5}
water.optimise(method=ade.methods.get_lmethod())

# dihedral constraints
butane = ade.Molecule(name='butane', smiles='CCCC')
butane.constraints.angular = {(0,1,2,3): 180}

butane.optimise(method=ade.methods.get_lmethod())