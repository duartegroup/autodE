import autode as ade

ade.Config.n_cores = 8


h2o = ade.Molecule(smiles="O")
na_ion = ade.Molecule(smiles="[Na+]")

# Initialise the [Na(H2O)3]+ complex and search 'conformers'
na_h2o_3 = ade.NCIComplex(na_ion, h2o, h2o, h2o)
na_h2o_3.find_lowest_energy_conformer(allow_connectivity_changes=True)

# Print .xyz files of all the generated conformers
for idx, conformer in enumerate(na_h2o_3.conformers):
    conformer.print_xyz_file(filename=f"conf{idx}.xyz")
