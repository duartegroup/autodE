import autode as ade

# Use ORCA DFT optimisations
ade.Config.lcode = ade.Config.hcode = 'orca'

if not ade.methods.ORCA().available:
    exit('This example requires an ORCA install')

# Locating transition states (TSs) in autodE requires defining a reaction.
# For example, the TS for a key step in a Beckmann rearrangement can be
# calculated with
r1 = ade.Reactant('Beckmann/reactant.xyz')
p1 = ade.Product('Beckmann/product.xyz')
p2 = ade.Product('Beckmann/water.xyz')

# Form the reaction and locate the transition state
rxn = ade.Reaction(r1, p1, p2)
rxn.locate_transition_state()

if rxn.ts is not None:
    print('TS has been found!')
    print('Imaginary frequency: ', rxn.ts.imaginary_frequencies[0])
