from autode import Reactant, Product, Reaction
from autode.input_output import xyz_file_to_atoms
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.locate_tss import get_ts

r = Reactant(name='cope_r', atoms=xyz_file_to_atoms('cope_r.xyz'))
p = Product(name='cope_p', atoms=xyz_file_to_atoms('cope_p.xyz'))
reaction = Reaction(r, p, name='cope')

# Define the bond rearrangement as tuples for each forming and breaking bond
bond_rearr = BondRearrangement(forming_bonds=[(5, 9)],
                               breaking_bonds=[(0, 1)])

ts = get_ts(reaction, r, bond_rearr)
print(ts.imaginary_frequencies)
