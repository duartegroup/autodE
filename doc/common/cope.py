import autode as ade
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.locate_tss import get_ts

ade.Config.n_cores = 8

# Use ORCA as both the high and low-level code
ade.Config.lcode = ade.Config.hcode = 'orca'

# and a smaller than default step size for a flat then quickly varying PES
ade.Config.max_step_size = 0.1
ade.Config.min_step_size = 0.02

r = ade.Reactant('cope_r.xyz')
p = ade.Product('cope_p.xyz')

# Define the bond rearrangement as tuples for each forming and breaking bond
bond_rearr = BondRearrangement(forming_bonds=[(5, 9)],
                               breaking_bonds=[(0, 1)])

ts = get_ts(name='cope', reactant=r, product=p, bond_rearr=bond_rearr)
print(ts.imaginary_frequencies)
