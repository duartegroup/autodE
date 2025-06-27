from molfunc import CoreMolecule, CombinedMolecule

fragments = {
    "NMe2": "CN([*])C",
    "NH2": "N[*]",
    "OH": "O[*]",
    "Me": "C[*]",
    "F": "F[*]",
}

methane = CoreMolecule(xyz_filename="CH4.xyz", atoms_to_del=[2])

for name, smiles in fragments.items():
    combined = CombinedMolecule(
        core_mol=methane, frag_smiles=smiles, name=f"CH3_{name}"
    )
    combined.print_xyz_file()
