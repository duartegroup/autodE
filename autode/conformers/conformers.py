from rdkit import Chem
from autode.atoms import Atom
from autode.config import Config
from autode.constants import Constants
from autode.geom import calc_rmsd
from autode.log import logger
import numpy as np


def get_atoms_from_rdkit_mol_object(rdkit_mol_obj, conf_id):
    """Generate atoms for conformers in rdkit_mol_obj

    Arguments:
        rdkit_mol_obj (rdkit.Chem.Mol): RDKit molecule
        conf_id (int): Conformer id to convert to atoms

    Returns:
        (list(autode.atoms.Atom)): Atoms
    """

    mol_block_lines = Chem.MolToMolBlock(rdkit_mol_obj,
                                         confId=conf_id).split('\n')
    mol_file_atoms = []

    # Extract atoms from the mol block
    for line in mol_block_lines:
        split_line = line.split()

        if len(split_line) == 16:
            atom_label = split_line[3]
            x, y, z = split_line[0], split_line[1], split_line[2]
            mol_file_atoms.append(Atom(atom_label, x=x, y=y, z=z))

    return mol_file_atoms


def get_unique_confs(conformers, energy_threshold_kj=1):
    """
    For a list of conformers return those that are unique based on an energy
    threshold in kJ mol^-1

    Arguments:
        conformers (list(autode.conformer.Conformer)):
        energy_threshold_kj (float): Energy threshold in kJ mol-1

    Returns:
        (list(autode.conformers.conformers.Conformer)): List of conformers
    """
    logger.info(f'Stripping conformers with energy ∆E < {energy_threshold_kj} '
                f'kJ mol-1 to others')

    n_conformers = len(conformers)

    # Conformer.energy is in Hartrees
    threshold = energy_threshold_kj / Constants.ha2kJmol

    # The first conformer must be unique, if it has an energy
    unique_conformers = []

    for conformer in conformers:

        if conformer.energy is None:
            logger.error('Conformer had no energy. Excluding')
            continue

        # Iterate through all the unique conformers already found and check
        # that the energy is not similar
        unique = True
        for other_conformer in unique_conformers:
            if np.abs(conformer.energy - other_conformer.energy) < threshold:
                unique = False
                break

        if unique:
            unique_conformers.append(conformer)

    n_unique_conformers = len(unique_conformers)
    logger.info(f'Stripped {n_conformers - n_unique_conformers} conformer(s) '
                f'from a total of {n_conformers}')

    if n_unique_conformers == 0:
        logger.error('Have no conformers!')

    return unique_conformers


def conf_is_unique_rmsd(conf, conf_list, rmsd_tol=None):
    """
    Determine if a conformer is unique based on an root mean squared
    displacement RMSD threshold

    Arguments:
        conf (autode.conformer.Conformer):
        conf_list (list((list(autode.conformer.Conformer)):

    Keyword Arguments:
        rmsd_tol (float): Tolerance for an equivalent structure based on the
                          rmsd in Å. If None then use the default value for
                          autode.Config.rmsd_threshold

    Returns:
        (bool):
    """
    rmsd_tol = Config.rmsd_threshold if rmsd_tol is None else rmsd_tol
    logger.info(f'Removing conformers with RMSD < {rmsd_tol} Å to any other')

    # Calculate the RMSD between this Conformer and the those in conf_list
    # using the Kabsch algorithm
    for other_conf in conf_list:
        conf_coords = conf.get_coordinates()
        other_conf_coords = other_conf.get_coordinates()

        if calc_rmsd(conf_coords, other_conf_coords) < rmsd_tol:
            return False

    return True
