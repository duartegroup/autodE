import autode as ade

ade.Config.n_cores = 2
ade.Config.hcode = "orca"

if not ade.methods.ORCA().is_available:
    exit("This example requires an ORCA install")

# Identity reactions require the reactants to be somehow distinguished from the products
# so that the breaking/forming bonds can be identified. For example, for
# H-H + H -> H + H-H
rxn = ade.Reaction(
    ade.Reactant(atoms=[ade.Atom("H", atom_class=1)], mult=2),
    ade.Reactant(atoms=[ade.Atom("H"), ade.Atom("H", x=0.8)]),
    ade.Product(atoms=[ade.Atom("H")], mult=2),
    ade.Product(atoms=[ade.Atom("H"), ade.Atom("H", atom_class=1, x=0.8)]),
)
rxn.calculate_reaction_profile()

print("∆E_r (kcal mol-1) = ", rxn.delta("E").to("kcal mol-1"))
print("∆E‡ (kcal mol-1)  = ", rxn.delta("E‡").to("kcal mol-1"))
print("TS imaginary freq = ", rxn.ts.imaginary_frequencies[0])
