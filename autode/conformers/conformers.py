from rdkit import Chem
from autode.log import logger
from autode.constants import Constants
from autode.atoms import Atom


def get_atoms_from_rdkit_mol_object(rdkit_mol_obj, conf_id):
    """Generate atoms for conformers in rdkit_mol_obj

        Arguments:
            rdkit_mol_obj (mol obj): Molecule object
            conf_id (int): Conformer ids to convert to atoms

        Returns:
            atoms (list(autode.atoms.Atom)):
        """

    mol_block_lines = Chem.MolToMolBlock(rdkit_mol_obj, confId=conf_id).split('\n')
    mol_file_atoms = []

    for line in mol_block_lines:
        split_line = line.split()
        if len(split_line) == 16:
            atom_label, x, y, z = split_line[3], split_line[0], split_line[1], split_line[2]
            mol_file_atoms.append(Atom(atom_label, x=float(x), y=float(y), z=float(z)))

    return mol_file_atoms


def get_unique_confs(conformers, energy_threshold_kj=1):
    """
    For a list of conformers return those that are unique based on an energy threshold

    Arguments:
        conformers (list(autode.conformer.Conformer)):
        energy_threshold_kj (float): Energy threshold in kJ mol-1

    Returns:
        (list(autode.conformers.conformers.Conformer)):
    """
    logger.info(f'Stripping conformers with energy ∆E < {energy_threshold_kj} kJ mol-1 to others')

    n_conformers = len(conformers)
    delta_e = energy_threshold_kj / Constants.ha2kJmol   # Conformer.energy is in Hartrees

    # The first conformer must be unique
    unique_conformers = conformers[:1]

    for i in range(1, n_conformers):

        if conformers[i].energy is None:
            logger.error('Conformer had no energy. Stripping')
            continue

        # Iterate through all the unique conformers already found and check that the energy is not similar
        unique = True
        for j in range(len(unique_conformers)):
            if conformers[i].energy - delta_e < conformers[j].energy < conformers[i].energy + delta_e:
                unique = False
                break

        if unique:
            unique_conformers.append(conformers[i])

    n_unique_conformers = len(unique_conformers)
    logger.info(f'Stripped {n_conformers - n_unique_conformers} conformer(s) from a total of {n_conformers}')

    return unique_conformers


