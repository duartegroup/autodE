import os

from typing import Collection, Sequence, Optional, Any, TYPE_CHECKING

from autode.atoms import Atom, Atoms
from autode.exceptions import XYZfileDidNotExist
from autode.exceptions import XYZfileWrongFormat
from autode.log import logger
from autode.utils import StringDict

if TYPE_CHECKING:
    from autode.species.molecule import Molecule
    from autode.species.species import Species


def xyz_file_to_atoms(filename: str) -> Atoms:
    """
    From a .xyz file get a list of autode atoms

    ---------------------------------------------------------------------------
    Arguments:
        filename: .xyz filename

    Raises:
        (autode.exceptions.XYZfileWrongFormat): If the file is the wrong format

    Returns:
        (autode.atoms.Atoms): Atoms
    """
    logger.info(f"Getting atoms from {filename}")

    _check_xyz_file_exists(filename)
    atoms = Atoms()
    n_atoms = 0

    for i, line in enumerate(open(filename, "r")):
        if i == 0:  # First line in an xyz file is the number of atoms
            n_atoms = _n_atoms_from_first_xyz_line(line)
            continue
        elif i == 1:  # Second line of an xyz file is the tittle line
            continue
        elif i == n_atoms + 2:
            break

        try:
            atom_label, x, y, z = line.split()[:4]
            atoms.append(Atom(atomic_symbol=atom_label, x=x, y=y, z=z))

        except (IndexError, TypeError, ValueError):
            raise XYZfileWrongFormat(
                f"Coordinate line {i} ({line}) not the correct format"
            )

    if len(atoms) != n_atoms:
        raise XYZfileWrongFormat(
            f"Number of atoms declared ({n_atoms}) "
            f"not equal to the number of atoms found "
            f"{len(atoms)}"
        )

    if n_atoms == 0:
        raise XYZfileWrongFormat(f"XYZ file ({filename}) had no atoms!")

    return atoms


def atoms_to_xyz_file(
    atoms: Collection[Atom],
    filename: str,
    title_line: str = "",
    append: bool = False,
):
    """
    Print a standard .xyz file from a list of atoms

    ---------------------------------------------------------------------------
    Arguments:
        atoms: List of autode atoms to print

        filename: Name of the file (with .xyz extension)

        title_line: Second line of the xyz file, can be blank

        append: Do or don't append to this file. With append=False
                filename will be overwritten if it already exists
    """
    assert atoms is not None
    assert filename.endswith(".xyz")

    with open(filename, "a" if append else "w") as xyz_file:
        print(len(atoms), title_line, sep="\n", file=xyz_file)

        for atom in atoms:
            x, y, z = atom.coord  # (Å)
            print(f"{atom.label:<3} {x:10.5f} {y:10.5f} {z:10.5f}", file=xyz_file)
    return None


def xyz_file_to_molecules(filename: str) -> Sequence["Molecule"]:
    """
    From a .xyz file containing potentially more than a single molecule
    return a list of molecules from it.

    ---------------------------------------------------------------------------
    Arguments:
        filename: Filename to open

    Returns:
        (list(autode.species.molecule.Molecule)): Molecules
    """
    from autode.species.molecule import Molecule  # prevents circular imports

    _check_xyz_file_exists(filename)

    lines = open(filename, "r").readlines()
    n_atoms = int(lines[0].split()[0])
    molecules = []

    for i in range(0, len(lines), n_atoms + 2):
        atoms = []
        title_line = StringDict(lines[i + 1])
        for j, line in enumerate(lines[i + 2 : i + n_atoms + 2]):
            symbol, x, y, z = line.split()[:4]
            atoms.append(Atom(atomic_symbol=symbol, x=x, y=y, z=z))

        molecule = Molecule(
            atoms=atoms, solvent_name=title_line.get("solvent", None)
        )

        _set_attr_from_title_line(molecule, "charge", title_line)
        _set_attr_from_title_line(molecule, "mult", title_line)
        _set_attr_from_title_line(
            molecule, "energy", title_line, key_in_line="E"
        )

        molecules.append(molecule)

    return molecules


def attrs_from_xyz_title_line(filename: str) -> StringDict:
    title_line = ""

    for i, line in enumerate(open(filename, "r")):
        if i == 1:
            title_line = line.strip()
            break

    return StringDict(title_line)


def _check_xyz_file_exists(filename: str) -> None:
    if not os.path.exists(filename):
        raise XYZfileDidNotExist(f"{filename} did not exist")

    if not filename.endswith(".xyz"):
        raise XYZfileWrongFormat("xyz file must have a .xyz file extension")

    return None


def _set_attr_from_title_line(
    species: "Species",
    attr: str,
    title_line: StringDict,
    key_in_line: Optional[str] = None,
) -> None:
    if key_in_line is None:
        key_in_line = attr  # Default to e.g. charge attribute is "charge = 0"
    try:
        setattr(species, attr, title_line[key_in_line])
    except IndexError:
        logger.warning(f"Failed to set the species {attr} from xyz file")

    return None


def _n_atoms_from_first_xyz_line(line: str) -> int:
    try:
        return int(line.strip())
    except (IndexError, ValueError):
        raise XYZfileWrongFormat("Number of atoms not found")
