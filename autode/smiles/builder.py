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
        return self.atoms is not None and len(self.queued_atoms) == 0

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
        if atoms is None or len(atoms) == 0:
            raise SMILESBuildFailed('Cannot build a structure with no atoms')

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

        # Add the first atom to the queue
        self.queued_atoms.append(0)
        return None

    def build(self, atoms, bonds):
        """
        Build a molecule by iterating through all the atoms adding it and
        each of it's neighbours. i.e.

        atoms = [C, H, H, H]

        1. Add C at origin
        2. Add all neighbours
        3. Done

        atoms = [C, C, C, 8xH]

        1. Add C at origin
        2. Add H3, C neighbours & update queued atoms to include the C
           that has been translated but needs it's neighbours adding to it

        ----------------------------------------------------------------------
        Arguments:
            atoms (list(autode.smiles.SMILESAtoms)):

            bonds (autode.smiles.SMILESBonds):
        """
        start_time = time()
        self._set_atoms_bonds(atoms, bonds)

        shifted_atoms = set()

        while not self.built:

            # Grab the last item to be added on the queue
            idx = self.queued_atoms[0]
            atom = self.atoms[idx]

            # Add all the atoms that are bonded to this one
            for bond in self.bonds.involving(idx):
                bonded_idx = bond[0] if bond[1] == idx else bond[1]

                if bonded_idx in shifted_atoms:
                    continue

                # Coordinate of this atom is the current position shifted by
                # the ideal distance in a direction of a empty coordination
                # site on the atom
                coord = bond.r0 * atom.type.empty_site() + atom.coord
                bonded_atom = self.atoms[bonded_idx]
                bonded_atom.translate(coord)

                # Atoms that are not terminal need to be added to the queue
                if not isinstance(atoms[bonded_idx].type, TerminalAtom):

                    # and the atom type rotated so an empty site is coincident
                    # with this atom
                    bonded_atom.type.rotate_onto(point=atom.coord,
                                                 coord=bonded_atom.coord)
                    # and queue
                    self.queued_atoms.append(bonded_idx)

            # And remove this atom from the queue and added to shifted
            self.queued_atoms.remove(idx)
            shifted_atoms.add(idx)
            logger.info(f'Queue: {self.queued_atoms}')

        logger.info(f'Built 3D in {(time() - start_time)*1E3:.2f} ms')
        return None

    def __init__(self):
        """
        Coordinate builder initialised from a set of atoms and bonds connecting
        them. This builder should generate something *reasonable* that can
        be cleaned up with a forcefield
        """
        self.atoms = None
        self.bonds = None

        # A queue of atom indexes the neighbours for which need to be added
        self.queued_atoms = []


class AtomType:

    def empty_site(self):
        """Iterator for the coordinate of the next free site"""
        return self._site_coords.pop(0)

    def rotate_onto(self, point, coord):
        """
        Rotate this atom type so the first site is parallel with a point
        and remove the site from the possibles

        Arguments:
            point (None | np.ndarray): shape = (3,)
            coord (np.ndarray): shape = (3,)

        """
        logger.info(f'Rotating atom at {coord} such that a vacant site is '
                    f'coincident to {point}')

        if point is None:   # No translation needed for no previous point
            return

        site = self._site_coords.pop(0)   # Currently vacant site

        v1 = site - coord
        v2 = point - coord

        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        arg = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # cosθ = v1.v2 / |v1||v2| := arg
        # cos(θ/2) = √(arg + 1) / √2
        # sin(θ/2) = √(1-arg) / √2
        a = np.sqrt(1.0 - arg) / np.sqrt(2)
        b, c, d = -normal * (np.sqrt(1 + arg) / np.sqrt(2))

        # 3D rotation matrix from the Euler–Rodrigues formula
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                               [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                               [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

        # Rotate all the sites (no need to translate as they're already
        # positioned around the origin)
        self._site_coords = [np.matmul(rot_matrix, site)
                             for site in self._site_coords]

        return None

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
        self._site_coords = [np.array([1.0, 0.0, 0.0])]


class LinearAtom(AtomType):

    def __init__(self):
        """
        Linear atom with sites pointing along the x-axis::

                    <---Atom--->
        """
        super().__init__()
        self._site_coords = [np.array([1.0, 0.0, 0.0]),
                             np.array([-1.0, 0.0, 0.0])]


class BentAtom(AtomType):

    def __init__(self):
        """
        Bent atom with sites generated by optimisation of H2O::

                       Atom
                     /     \
        """
        super().__init__()
        self._site_coords = [np.array([-0.78226654, -0.62294387, 0.0]),
                             np.array([0.78322832, -0.62173419, 0.0])]


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
        self._site_coords = [np.array([-0.580775, -0.75435372, -0.30602419]),
                             np.array([-0.404709,  0.86798519, -0.28777090]),
                             np.array([0.0763827, -0.01927872,  0.99689218]),
                             np.array([0.9089159, -0.09390161, -0.40626889])]
