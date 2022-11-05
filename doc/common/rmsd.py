#!/usr/bin/env python3
from autode.species import Species
from autode.geom import calc_rmsd
from autode.input_output import xyz_file_to_atoms
from shutil import copyfile
import os
import argparse

folder_name = "unique_conformers"


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames", action="store", nargs="+", help=".xyz files to compare"
    )

    parser.add_argument(
        "-t",
        action="store",
        default=1.0,
        type=float,
        help="RMSD threshold (Å)",
    )

    parser.add_argument(
        "-oh",
        action="store_true",
        default=False,
        help="Only compare heavy atoms",
    )

    return parser.parse_args()


def get_molecules_no_hydrogens(molecules):
    """Remove all hydrogen atoms from  a list of molecules"""

    no_h_molecules = []

    for molecule in molecules:
        molecule.atoms = [atom for atom in molecule.atoms if atom.label != "H"]

        no_h_molecules.append(molecule)

    return no_h_molecules


def get_and_copy_unique_confs(xyz_filenames, only_heavy_atoms, threshold_rmsd):
    """For each xyz file generate a species and copy it to a folder if it is
    unique based on an RMSD threshold"""

    molecules = [
        Species(
            name=fn.rstrip(".xyz"),
            atoms=xyz_file_to_atoms(fn),
            charge=0,
            mult=1,
        )
        for fn in xyz_filenames
    ]

    if only_heavy_atoms:
        molecules = get_molecules_no_hydrogens(molecules)

    unique_mol_ids = []

    for i in range(len(molecules)):
        mol = molecules[i]

        is_unique = True

        for j in unique_mol_ids:
            rmsd = calc_rmsd(mol.coordinates, molecules[j].coordinates)

            if rmsd < threshold_rmsd:
                is_unique = False
                break

        if is_unique:
            unique_mol_ids.append(i)

    print("Number of unique molecules = ", len(unique_mol_ids))

    # Copy all the unique .xyz files to a new folder
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for i in unique_mol_ids:
        xyz_filename = xyz_filenames[i]
        copyfile(xyz_filename, os.path.join(folder_name, xyz_filename))

    return None


if __name__ == "__main__":

    args = get_args()
    assert all(fn.endswith(".xyz") for fn in args.filenames)

    get_and_copy_unique_confs(
        xyz_filenames=args.filenames,
        only_heavy_atoms=args.oh,
        threshold_rmsd=args.t,
    )
