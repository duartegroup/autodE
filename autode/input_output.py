import os
from autode.atoms import Atom
from autode.exceptions import XYZfileDidNotExist
from autode.exceptions import XYZfileWrongFormat
from autode.log import logger


def xyz_file_to_atoms(filename):
    """
    From an .xyz file get a list of atoms

    Arguments:
        filename (str): .xyz filename

    Returns:
        (list(autode.atoms.Atom)): Atoms
    """
    logger.info(f'Getting atoms from {filename}')

    atoms = []

    if not os.path.exists(filename):
        raise XYZfileDidNotExist

    if not filename.endswith('.xyz'):
        raise XYZfileWrongFormat

    # Open the file that exists and should(!) be in the correct format
    with open(filename, 'r') as xyz_file:

        try:
            # First item in an xyz file is the number of atoms
            n_atoms = int(xyz_file.readline().split()[0])

        except IndexError:
            raise XYZfileWrongFormat

        # XYZ lines should be the following 2 + n_atoms lines
        xyz_lines = xyz_file.readlines()[1:n_atoms+1]

        for line in xyz_lines:

            try:
                atom_label, x, y, z = line.split()[:4]
                atoms.append(Atom(atomic_symbol=atom_label, x=x, y=y, z=z))

            except (IndexError, TypeError, ValueError):
                raise XYZfileWrongFormat

    return atoms


def atoms_to_xyz_file(atoms, filename, title_line='', append=False):
    """
    Print a standard .xyz file from a list of atoms

    Arguments:
        atoms (list(autode.atoms.Atom)): 
        filename (str):

    Keyword Arguments:
        title_line (str):
        append (bool):
    """

    with open(filename, 'a' if append else 'w') as xyz_file:
        print(len(atoms), title_line, sep='\n', file=xyz_file)

        for atom in atoms:
            x, y, z = atom.coord
            print(f'{atom.label:<3}{x:^10.5f}{y:^10.5f}{z:^10.5f}',
                  file=xyz_file)
    return None
