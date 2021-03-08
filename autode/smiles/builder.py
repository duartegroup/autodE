import numpy as np
from time import time
from autode.log import logger
from autode.bonds import get_avg_bond_length
from autode.exceptions import SMILESBuildFailed
from autode.smiles.base import SMILESAtom, SMILESBond


class Builder:
    """3D geometry builder

    Atoms:  C, 4H               H  H
    Bonds:  4 x C-H      -->     C
                               H  H
    """

    @property
    def built(self):
        """Have all the atoms been shifted appropriately"""
        return len(self.shifted_idxs) == len(self.atoms)

    def _explicit_all_hydrogens(self):
        """Convert all implicit hydrogens to explicit ones"""

        h_atoms = []
        for idx, atom in enumerate(self.atoms):

            if not hasattr(atom, 'n_hydrogens') or atom.n_hydrogens is None:
                raise SMILESBuildFailed('All atoms must have a defined number '
                                        f'of hydrogens to build. {atom} had '
                                        f'n_hydrogens = None')

            for _ in range(atom.n_hydrogens):
                h_atoms.append(SMILESAtom('H', n_hydrogens=0))

                # Add the bond between the current atom and the new H
                h_idx = len(self.atoms) + len(h_atoms) - 1
                self.bonds.append(SMILESBond(idx, h_idx, symbol='-'))

            # zero the number of implicit hydrogens bonded to this atom now
            # they are explicit
            atom.n_hydrogens = 0

        self.atoms += h_atoms
        return

    def _set_atoms_bonds(self, atoms, bonds):
        """
        From a list of SMILESAtoms, and SMILESBonds set the required attributes
        and convert all implicit hydrogens into explicit atoms
        """
        self.atoms, self.bonds = atoms, bonds
        self._explicit_all_hydrogens()

        # Set the ideal bond lengths
        for bond in self.bonds:
            idx_i, idx_j = bond
            bond.r0 = get_avg_bond_length(self.atoms[idx_i].label,
                                          self.atoms[idx_j].label)

        # Set the atom types for each atom, e.g. tetrahedral etc.
        for i, atom in enumerate(self.atoms):

            # Ensure every atom is initialised at the origin
            atom.coord = np.zeros(3)

            # Atom type is determined by the number of bonded atoms, and the
            # 'hybridisation' as well as the stereochemistry
            n_bonded = bonds.n_involving(i)

            if n_bonded == 1:                                 # e.g. H2, FCH3
                atom.type = TerminalAtom()

            elif n_bonded == 2:                               # e.g. OH2, SR2
                if atom.label in atoms.chalcogens:
                    atom.type = BentAtom()

                else:                                         # e.g. AuR2
                    atom.type = LinearAtom()

            elif n_bonded == 3:                               # e.g. NH3
                raise NotImplementedError

            elif n_bonded == 4:                               # e.g. CH4
                atom.type = TetrahedralAtom()

            else:
                raise NotImplementedError

        return None

    def non_shifted_atoms_idxs(self):
        """Iterator over all atoms that have yet to be shifted"""

        if self.built:
            return

        for i, atom in enumerate(self.atoms):
            if i not in self.shifted_idxs:
                yield i, atom

    def build(self, atoms, bonds):
        """
        Build a molecule

        Arguments:
            atoms (list(autode.smiles.SMILESAtoms)):

            bonds (autode.smiles.SMILESBonds):
        """
        start_time = time()
        self._set_atoms_bonds(atoms, bonds)

        prev_position = None

        for i, atom in self.non_shifted_atoms_idxs():

            atom.translate(vec=self.position)
            atom.type.rotate_onto(prev_position)
            self.shifted_idxs.add(i)

            # Add all the atoms that are bonded to this one
            empty_sites = atom.type.empty_sites()

            for bond in self.bonds.involving(i):
                bonded_idx = bond[0] if bond[1] == i else bond[1]

                if bonded_idx in self.shifted_idxs:
                    continue

                coord = bond.r0 * next(empty_sites) + self.position

                self.atoms[bonded_idx].translate(coord)
                self.shifted_idxs.add(bonded_idx)

            # Update the positions, at 0 if there are no more atoms to add
            # from this one
            prev_position = np.array(self.position, copy=True)
            self.position = next(empty_sites, np.zeros(3))

        logger.info(f'Built 3D in {(time() - start_time)*1E3:.2f} ms')
        return None

    def __init__(self, ):
        """
        Coordinate builder initialised from a set of atoms and bonds connecting
        them. This builder should generate something *reasonable* that can
        be cleaned up with a forcefield
        """
        self.atoms = None
        self.bonds = None

        # Initial position that will be updated as atoms are added
        self.position = np.zeros(3)

        # Atoms that have been shifted from the origin
        self.shifted_idxs = set()


class AtomType:

    def empty_sites(self):
        """Iterator for the coordinate of the next free site"""
        for coord in self._site_coords:
            yield coord

    def rotate_onto(self, point):
        """
        Rotate this atom type so the first site is parallel with a point
        and remove the site from the possibles

        Arguments:
            point (None | np.ndarray): shape = (3,)

        """
        if point is None:   # No translation needed for no previous point
            return

        raise NotImplementedError

    def __init__(self):
        """Base atom type class

        _site_coords should contain a list of unit vectors pointing in
        directions where other atoms can be added
        """

        self._site_coords = []


class TerminalAtom(AtomType):

    def __init__(self):
        """
        Terminal atom with a site pointing along the x-axis::

                    Atom--->
        """
        super().__init__()
        self._site_coords = np.array([[1.0, 0.0, 0.0]])


class LinearAtom(AtomType):

    def __init__(self):
        """
        Linear atom with sites pointing along the x-axis::

                    <---Atom--->
        """
        super().__init__()
        self._site_coords = np.array([[1.0, 0.0, 0.0],
                                      [-1.0, 0.0, 0.0]])


class BentAtom(AtomType):

    def __init__(self):
        """
        Bent atom with sites generated by optimisation of H2O::

                       Atom
                     /     \
        """
        super().__init__()
        self._site_coords = np.array([[-0.78226654, -0.62294387, 0.0],
                                      [0.78322832, -0.62173419, 0.0]])


class TetrahedralAtom(AtomType):

    def __init__(self):
        """
        Tetrahedral atom with sites generated by optimisation of methane
        (then normalising the CH distances to 1 Å)::

                         |  /
                       Atom
                     /     \
        """
        super().__init__()
        self._site_coords = np.array([[-0.580775, -0.75435372, -0.30602419],
                                      [-0.404709,  0.86798519, -0.28777090],
                                      [0.0763827, -0.01927872,  0.99689218],
                                      [0.9089159, -0.09390161, -0.40626889]])
