import autode as ade

xtb, orca = ade.methods.XTB(), ade.methods.ORCA()

if not (orca.is_available and xtb.is_available):
    exit("This example requires both an ORCA and XTB install")

# Dinitrogen molecule
n2 = ade.Molecule(smiles="N#N")

# For both XTB and ORCA optimise to a minimum and calculate a numerical Hessian
for method in (xtb, orca):

    n2.optimise(method=method)
    n2.calc_hessian(
        method=method, numerical=True, use_central_differences=True
    )

    print(f"Numerical frequency at {method.name}:", n2.vib_frequencies)
