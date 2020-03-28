from autode.log import logger
from autode.exceptions import XYZfileWrongFormat, XYZfileDidNotExist
from autode.atoms import Atom
import os


def xyz_file_to_atoms(filename):
    """/
    From an .xyz file get a list of atoms

    Arguments:
        filename (str): .xyz filename

    Returns:
        (list(autode.atoms.Atom)):
    """
    logger.info(f'Getting atoms from {filename}')

    atoms = []

    if not os.path.exists(filename):
        raise XYZfileDidNotExist

    if not filename.endswith('.xyz'):
        raise XYZfileWrongFormat

    # Open the file that exists and should(!) be in the correct format
    with open(filename, 'r') as xyz_file:
        n_atoms = int(xyz_file.readline().split()[0])       # First item in an xyz file is the number of atoms
        xyz_lines = xyz_file.readlines()[1:n_atoms+1]       # XYZ lines should be the following 2 + n_atoms lines

        for line in xyz_lines:

            try:
                atom_label, x, y, z = line.split()[:4]
                atoms.append(Atom(atomic_symbol=atom_label, x=float(x), y=float(y), z=float(z)))

            except (IndexError, TypeError):
                raise XYZfileWrongFormat

    return atoms
