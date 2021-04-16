import numpy as np
from autode.log import logger
from autode.atoms import Atom
from autode.exceptions import InvalidSmilesString

bond_order_symbols = ['-', '=', '#', '$']
organic_symbols = ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
aromatic_symbols = ['b', 'c', 'n', 'o', 's', 'p']


class SMILESAtom(Atom):
    """Atom in a SMILES string"""

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'SMILESAtom({self.label}, stereo={self.stereochem})'

    @property
    def is_shifted(self):
        """Has this atom been shifted from the origin?"""
        return False if np.allclose(self.coord, np.zeros(3)) else True

    @property
    def is_aromatic(self):
        """Is this atom 'aromatic'?"""
        return self.smiles_label in aromatic_symbols

    @property
    def has_stereochem(self):
        """Does this atom have associated stereochemistry?"""
        return self.stereochem is not None

    @property
    def n_bonded(self):
        """How many atoms are bonded to this one?"""
        return 0 if self.neighbours is None else len(self.neighbours)

    def invert_stereochem(self):
        """Invert the stereochemistry at this centre"""
        logger.info('Inverting stereochemistry..')

        if self.stereochem == '@':
            self.stereochem = '@@'

        elif self.stereochem == '@@':
            self.stereochem = '@'

        else:
            logger.info('No change needed')

        return

    def __init__(self, label, stereochem=None, n_hydrogens=None, charge=0):
        """
        SMILES atom initialised at the origin

        ----------------------------------------------------------------------
        Arguments:
            label (str): Label / atomic symbol of this atom

        Keyword Arguments:
            n_hydrogens (int | None): Number of hydrogens, None means unset and
                                      should be determined implicitly

            stereochem (str | None):

            charge (int): Formal charge on this atom
        """
        super().__init__(atomic_symbol=label.capitalize())

        # SMILES label may be distinct from the atom label, e.g. aromatic atoms
        self.smiles_label = label

        self.charge = charge
        self.n_hydrogens = n_hydrogens
        self.stereochem = stereochem

        # ---------- Attributes used for building the 3D structure ----------
        self.type = None
        self.neighbours = None
        self.in_ring = False
        self.is_pi = False if label not in aromatic_symbols else True


class SMILESBond:
    """Bond in a SMILES string"""

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'SMILESBond({self._list}, order={self.order})'

    def __getitem__(self, item):
        return self._list[item]

    @property
    def atom_indexes(self):
        """Atom indexes for the atoms in this bond"""
        return {self._list[0], self._list[1]}

    def is_cis(self, atoms):
        """Is this bond a cis double bond?"""
        i, j = self._list

        if atoms[i].stereochem is None:
            return False

        return self.order == 2 and atoms[i].stereochem == atoms[j].stereochem

    def is_trans(self, atoms):
        """Is this bond a trans double bond?"""
        return self.order == 2 and not self.is_cis(atoms)

    def distance(self, atoms):
        """Distance of this bond (Å) given a set of atoms"""
        idx_i, idx_j = self._list
        return np.linalg.norm(atoms[idx_i].coord - atoms[idx_j].coord)

    @property
    def symbol(self):
        """SMILES symbol for this bond e.g. # for a triple bond"""
        return bond_order_symbols[self.order - 1]

    @symbol.setter
    def symbol(self, value):
        """Allow for a symbol to be set, keeping track of only the order"""
        self.order = bond_order_symbols.index(value) + 1

    def __init__(self, idx_i: int, idx_j: int, symbol: str):
        """
        Bond between two atoms from a SMILES string, sorted from low to high

        Arguments:
            idx_i (int):
            idx_j (int):
            symbol (str): Bond order symbol
        """
        self._list = [idx_i, idx_j]

        if symbol not in bond_order_symbols:
            raise InvalidSmilesString(f'{symbol} is an unknown bond type')

        self.in_ring = False
        self.closes_ring = False
        self.order = bond_order_symbols.index(symbol) + 1

        self.r0 = None                               # Ideal bond distance (Å)


class RingBond(SMILESBond):
    """Dangling bond created when a ring is found"""

    def __repr__(self):
        return f'RingSMILESBond({self._list}, order={self.order})'

    def close(self, idx, symbol):
        """Close this bond using an atom index"""
        self._list = list(sorted([self[0], idx]))

        # Only override implicit single bonds with double, triple etc.
        if self.symbol == '-':
            self.symbol = symbol

        return None

    def __init__(self, idx_i, symbol, bond_idx=None):
        """Initialise the bond with a non-existent large index

        Arguments:
            idx_i (int): Index of one atom in this bond

            symbol (str): Symbol of this bond, in bond_order_symbols

        Keyword Arguments:
            bond_idx (None | int): Index for this bond in the bond list
        """
        super().__init__(idx_i=idx_i, idx_j=99999, symbol=symbol)

        self.in_ring = True
        self.closes_ring = True
        self.bond_idx = bond_idx


class SMILESBonds(list):

    def _bond_exists(self, bond):
        """Does this bond already exist in this set of bonds?"""
        return any(bond.atom_indexes == item.atom_indexes for item in self)

    def n_involving(self, idx):
        """How many bonds does an atom (given as a index) have?"""
        return len(self.involving(idx))

    def involving(self, *args):
        """Get all the bonds involving a particular atom (given as a index)

        Arguments:
            args (int):

        Returns:
            (list(autode.smiles.SMILESBond)):
        """
        idxs = set(args)

        return [bond for bond in self if idxs.issubset(set(bond.atom_indexes))]

    def first_involving(self, *args):
        """First bond that includes some atom indexes"""
        idxs = set(args)
        return next(b for b in self if idxs.issubset(set(b.atom_indexes)))

    def append(self, bond: SMILESBond):
        """Add another SMILESBond to this list"""

        if self._bond_exists(bond) or len(set(bond.atom_indexes)) != 2:
            return

        return super().append(bond)

    def insert(self, index: int, bond: SMILESBond):
        """Insert a bond into this list if it does not already exist"""

        if self._bond_exists(bond) or len(set(bond.atom_indexes)) != 2:
            return

        return super().insert(index, bond)
