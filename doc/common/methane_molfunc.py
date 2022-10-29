from molfunc import CoreMolecule, CombinedMolecule

fragments = {
    "NMe2": "CN([Fr])C",
    "NH2": "N[Fr]",
    "OH": "O[Fr]",
    "Me": "C[Fr]",
    "F": "F[Fr]",
}

methane = CoreMolecule(xyz_filename="CH4.xyz", atoms_to_del=[2])

for name, smiles in fragments.items():

    combined = CombinedMolecule(
        core_mol=methane, frag_smiles=smiles, name=f"CH3_{name}"
    )
    combined.print_xyz_file()
