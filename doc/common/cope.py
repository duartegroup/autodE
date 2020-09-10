from autode import Reactant, Product, Reaction
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.locate_tss import get_ts

r = Reactant('cope_r.xyz')
p = Product('cope_p.xyz')
reaction = Reaction(r, p, name='cope')

# Define the bond rearrangement as tuples for each forming and breaking bond
bond_rearr = BondRearrangement(forming_bonds=[(5, 9)],
                               breaking_bonds=[(0, 1)])

ts = get_ts(reaction, r, bond_rearr)
print(ts.imaginary_frequencies)
