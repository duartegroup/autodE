"""
(Open)SMILES parser implemented based on

1. http://opensmiles.org/
2. https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system

as of 03/2021
"""
from time import time
from autode.log import logger
from autode.atoms import Atom
from autode.atoms import elements
from autode.exceptions import InvalidSmilesString
from autode.smiles.base import (SMILESAtom, SMILESBond, SMILESBonds, RingBond,
                                aromatic_symbols,
                                organic_symbols,
                                bond_order_symbols)


class Parser:

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def n_bonds(self):
        return len(self.bonds)

    @property
    def charge(self):
        """Return the total charge on all the atoms"""
        return sum(atom.charge for atom in self.atoms)

    @property
    def canonical_atoms(self):
        """Generate canonical autodE atoms from this set"""
        if self.n_atoms == 0:
            raise ValueError('Had no atoms - cannot return a canonical set')

        atoms = []
        for atom in self.atoms:
            x, y, z = atom.coord
            atoms.append(Atom(atom.label, x=x, y=y, z=z))

        return atoms

    @property
    def mult(self):
        """Approximate spin multiplicity (2S+1). For multiple unpaired
        electrons will default to a singlet"""

        n_electrons = (sum([atom.atomic_number for atom in self.atoms])
                       - self.charge)

        return (n_electrons % 2) + 1

    @property
    def parsed(self):
        """Has the parser parsed every character of the SMILES string"""
        return len(self.parsed_idxs) == len(self._string)

    def _check_smiles(self):
        """Check the SMILES string for unsupported characters"""
        unsupported_chars = [':', '.', '*', '%']
        if any(char in self._string for char in unsupported_chars):
            raise InvalidSmilesString(f'{self._string} had invalid characters')

    @property
    def smiles(self):
        return self._string

    @smiles.setter
    def smiles(self, string):
        """Set the SMILES string for the parser and reset"""
        self._string = str(string.strip())  # strip leading/trailing whitespace
        self._check_smiles()

        # Reset all the defaults for the parser
        self.parsed_idxs = set()
        self.atoms = []
        self.bonds = SMILESBonds()

    def _parse_sq_bracket(self, string):
        """
        Parse a section in a square bracket

        e.g. [C], [CH3], [Cu+2], [O-], [C@H]
        """
        if '(' in string or ')' in string:
            raise InvalidSmilesString('Cannot parse branch in "[]" section')

        if len(string) == 0:
            raise InvalidSmilesString('"[]" must contain something')

        elif len(string) == 1:
            # Single element e.g. [C] i.e. string = 'C'
            self.atoms.append(SMILESAtom(string, n_hydrogens=0))
            return

        # e.g. [Cu++], first two characters are an element
        if string[:2] in elements:
            label, rest = string[:2], string[2:]

        # e.g. [CH2] or [n+]
        elif string[0] in elements or string[0] in aromatic_symbols:
            label, rest = string[0], string[1:]

        # e.g. [999C]  NOTE: SMILES does allow for isotopes, but they're not
        # supported
        else:
            raise InvalidSmilesString(f'Unknown first item {string} in a "[]"')

        if len(rest) == 0:  # e.g. [Cu] etc.
            self.atoms.append(SMILESAtom(label, n_hydrogens=0))
            return

        if any(elem in rest for elem in elements if elem != 'H'):
            raise InvalidSmilesString(f'Only expecting hydrogens in {rest}')

        atom = SMILESAtom(label=label,
                          n_hydrogens=atomic_n_hydrogens(rest),
                          charge=atomic_charge(rest),
                          stereochem=atomic_sterochem(rest))

        self.atoms.append(atom)
        return None

    def _parse_next_sq_bracket(self, idx):
        """
        Parse the next square bracket section from the SMILES e.g.

        CCC [ CH3]
           ^
           |
          idx

        Arguments:
            idx (int): Position in the SMILES string for the [
        """
        if idx == len(self.smiles) - 1:
            raise InvalidSmilesString('"[" cannot appear at '
                                      'the end of a SMILES string')

        # Split the on closed square brackets e.g
        # [C -> ['C']   [C] -> ['C', '']   [CH4] -> ['CH4', '']
        closing_brackets_sec = self.smiles[idx + 1:].split(']')

        if len(closing_brackets_sec) == 1:
            raise InvalidSmilesString('Bracket "]" not closed')

        # [C] -> 'C',  [CH4] -> 'CH4'
        bracketed_sec = self.smiles[idx + 1:].split(']')[0]
        n_bracket_chars = len(bracketed_sec)
        self._parse_sq_bracket(bracketed_sec)

        # Have now parsed i+1 -- n_bracket_chars+1 inclusive
        # where the +1 is from the final ]
        self.parsed_idxs.update(list(range(idx, idx + n_bracket_chars + 2)))
        return None

    def _add_bond(self, symbol, idx, prev_atom_idx=None):
        """
        Add a bond to the list of bonds from the previously added atom to

        Arguments:
            symbol (str): Symbol of this bond e.g. # for a double bond, see
                          bond_order_symbols

            idx (int): Index of the position in the SMILES string

        Keyword Arguments:
            prev_atom_idx (int | None): Index to bond the added atom to
        """
        if self.n_atoms == 1:               # First atom, thus no bonds to add
            return

        if prev_atom_idx is None:
            prev_atom_idx = self.n_atoms - 2

        self.bonds.append(SMILESBond(prev_atom_idx, self.n_atoms-1,
                                     symbol=symbol))

        if symbol == '=':
            self._set_double_bond_stereochem(idx)
        return None

    def _set_double_bond_stereochem(self, idx):
        """
        Set the stereochemistry for the atoms involved in a double bond (E/Z
        or cis/trans) that has just been added to the system e.g.::

            C(/F)=C/F
                  ^
                  |
                 idx

        where the slashes refer to the "up-ness" or "down-ness" of each single
        bond is relative to the carbon atom

        Arguments:
            idx (int): Index of the current position in the SMILES string
        """
        if '/' not in self._string and '\\' not in self._string:
            # No defined double bond setereochemistry
            return

        # Index that has been added previously and the new one
        atom_idx_j, atom_idx_i = self.bonds[-1]

        # Now set the up or down-ness of the atoms that are bonded with a
        # double bond, with respect to the next (or previous) atom
        for char in self._string[idx:]:
            if char == '/':
                self.atoms[atom_idx_i].stereochem = 'al_up'
                break

            if char == '\\':
                self.atoms[atom_idx_i].stereochem = 'al_down'
                break

        # Parse backwards from the final atom to assign the stereochemistry of
        # atom_j. Needs to allow for branching e.g. C(\F)=C/F is trans
        branched = False

        for char in self._string[:idx][::-1]:
            if char == ')':                                # Generated a branch
                branched = True

            if char == '(':                                # Closed a branch
                branched = False

            if char == '\\':
                stereo = 'al_up' if not branched else 'al_down'
                self.atoms[atom_idx_j].stereochem = stereo
                break

            if char == '/':
                stereo = 'al_down' if not branched else 'al_up'
                self.atoms[atom_idx_j].stereochem = stereo
                break

        return None

    def _set_implicit_hs(self):
        """
        Given a completely parsed set of atoms from a SMILES string set the
        implicit hydrogens for all atoms where they're defined. From ref [1]
        elems_pos_val is defined

        NOTE: Elements with implicit hydrogens must be neutral
        """
        elems_poss_val = {'B': (3,), 'C': (4,), 'N': (3, 5), 'O': (2,),
                          'P': (3, 5), 'S': (2, 4, 6), 'F': (1,), 'Cl': (1,),
                          'Br': (1,), 'I': (1,)}

        for idx, atom in enumerate(self.atoms):

            # Only consider atoms with undefined number of hydrogens
            if atom.n_hydrogens is not None:
                continue

            if atom.label not in elems_poss_val.keys():
                raise InvalidSmilesString('Could not define implicit hydrogens'
                                          f'for {atom.label}')

            bonds = self.bonds.involving(idx)
            sum_bond_orders = sum(bond.order for bond in bonds)

            # If the sum of the bond order is less than the minimum valance
            # then add the appropriate number of hydrogens to satisfy the
            # implicit valance
            for valance in elems_poss_val[atom.label]:
                if sum_bond_orders <= valance:
                    atom.n_hydrogens = valance - sum_bond_orders
                    break

                atom.n_hydrogens = 0

        return None

    def parse(self, smiles: str):
        """
        Parse a SMILES string e.g. '[He]', 'C'
        """
        self.smiles = smiles
        logger.info(f'Parsing {self.smiles}')

        start_time = time()
        branch_idxs = []     # Indexes of branch points
        unclosed_bonds = {}  # Bonds that must be closed
        prev_idx = None      # Index of the previous atom to bond the next to

        # Enumerate over the string until all characters have been parsed
        for i, char in enumerate(self._string):

            # Determine the type of bond the next added atom is bonded with
            if i > 0 and self._string[i-1] in bond_order_symbols:
                bond_symbol = self._string[i-1]         # double, triple etc.
            else:
                bond_symbol = '-'                       # single bonds implicit

            # Skip any parsed atoms, bond order chars and cis/trans definitions
            if i in self.parsed_idxs or char in bond_order_symbols+['/', '\\']:
                continue

            # Integer for a dangling bond e.g. C1, C=1, N3 etc.
            elif char.isdigit():
                ring_idx = int(char) - 1

                # This bond is in the dictionary and can be closed and removed
                if ring_idx in unclosed_bonds.keys():
                    ring_bond = unclosed_bonds.pop(ring_idx)
                    ring_bond.close(self.n_atoms-1, symbol=bond_symbol)
                    self.bonds.append(ring_bond)
                    continue

                unclosed_bonds[ring_idx] = RingBond(idx_i=self.n_atoms-1,
                                                    symbol=bond_symbol)

            # Any square bracketed atom with hydrogens defined e.g. [OH], [Fe]
            elif char == '[':
                self._parse_next_sq_bracket(idx=i)

            elif char == '(':                                      # New branch
                if i != 0 and self._string[i-1] == ')':
                    # Directly opened a new branch so keep the previous index
                    pass
                else:
                    branch_idxs.append(self.n_atoms - 1)
                continue

            elif char == ')':                                   # Closed branch
                if len(branch_idxs) == 0:
                    raise InvalidSmilesString('Closed unopened bracket "("')

                # If the next character is another branch from the same atom
                # then the branch index should not be deleted
                prev_idx = branch_idxs[-1]

                if next_char(self._string, i) != '(':
                    del branch_idxs[-1]

                continue

            # only Cl, Br
            elif char + next_char(self._string, i) in ('Cl', 'Br'):
                atom = SMILESAtom(label=char + self._string[i + 1])
                self.atoms.append(atom)
                # Have also parsed the next character
                self.parsed_idxs.update([i, i+1])

            # e.g. C, B, O
            elif char in organic_symbols + aromatic_symbols:
                self.atoms.append(SMILESAtom(label=char))

            else:
                raise InvalidSmilesString(f'Unsupported character {char}')

            # Finally add the bond and add this character to those parsed
            self._add_bond(bond_symbol, idx=i, prev_atom_idx=prev_idx)
            self.parsed_idxs.add(i)

            # Reset the index of the previous atom, so the next atom
            # will be bonded to the previously added one (unless a branch has
            # been closed)
            prev_idx = self.n_atoms - 1

        if len(unclosed_bonds) > 0:
            raise InvalidSmilesString('Found unclosed rings')

        self._set_implicit_hs()
        logger.info(f'Parsed SMILES in {(time() - start_time)*1E3:.2f} ms')
        return None

    def __init__(self):
        """SMILES Parser"""

        self._string = ''

        # Indexes of the characters in the SMILES string that have been parsed
        self.parsed_idxs = set()
        self.atoms = []
        self.bonds = SMILESBonds()


def atomic_charge(string):
    """
    Parse a section of a SMILES string associated with an atom for the
    formal charge on the atom, will ignore anything but +, -. e.g.

    +   ->   1
    -   ->  -1
    ++  ->   2
    H+  ->   1

    Returns:
        (int): charge
    """
    charge = 0
    for i, item in enumerate(string):

        if item == '+':
            sign = 1

        elif item == '-':
            sign = -1

        else:  # Not a charge determining portion
            continue

        # +3 or +2 or -2 etc.
        if next_char(string, i).isdigit():
            return sign * int(string[i + 1])

        # ++  or --
        elif next_char(string, i) in ('+', '-'):
            return sign * 2

        # just - or +, thus the charge is just the sign
        else:
            return sign

    return charge


def atomic_sterochem(string):
    """
    Extract the first occurring atomic stereochemistry from a partial
    SMILES i.e

    @H3  ->  @
    @    ->  @
    @@-  ->  @@

    Arguments:
        string (str):

    Returns:
        (str | None): Type of point stereochemistry
    """
    for i, item in enumerate(string):

        if item == '@':
            if next_char(string, i) == '@':
                return '@@'

            return '@'

    return None


def atomic_n_hydrogens(string):
    """
    Extract the number of hydrogens from a partial SMILES, i.e.

    H3-  ->  3
    H    ->  1
    C    ->  0

    Arguments:
        string (str):

    Returns:
        (int): Number of hydrogens
    """
    for i, item in enumerate(string):

        if item == 'H':
            # e.g. [CH3]  where rest = H3  or [OH-]
            if next_char(string, i).isdigit():
                return int(string[i + 1])

            # e.g. [OH]
            else:
                return 1

    return 0


def next_char(string, idx):
    """
    Get the next character in a string if it exists otherwise return
    an empty string

    Arguments:
        string (str):
        idx (idx): Index of the current position in the string

    Returns:
        (str):
    """
    if idx >= len(string) - 1:
        return ''

    return string[idx + 1]
