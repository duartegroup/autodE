import numpy as np
from typing import Union
from autode.log import logger
from autode.values import Distance, Angle
from autode.geom import get_rot_mat_euler


class AtomCollection:

    @property
    def n_atoms(self):
        """Number of atoms in this set"""
        return 0 if self.atoms is None else len(self.atoms)

    @property
    def coordinates(self):
        """Numpy array of coordinates"""
        if self.atoms is None:
            return None

        return np.array([a.coord for a in self.atoms], dtype='f8', copy=True)

    @coordinates.setter
    def coordinates(self,
                    value: np.ndarray):
        """Set the coordinates from a numpy array

        Arguments:
            value (np.ndarray): Shape = (n_atoms, 3) or (3*n_atoms) as a
                                row major vector
        """
        if self.atoms is None:
            raise ValueError('Must have atoms set to be able to set the '
                             'coordinates of them')

        if value.ndim == 1:
            assert value.shape == (3 * self.n_atoms,)
            value = value.reshape((-1, 3))

        elif value.ndim == 2:
            assert value.shape == (self.n_atoms, 3)

        else:
            raise AssertionError('Cannot set coordinates from a array with'
                                 f'shape: {value.shape}. Must be 1 or 2 '
                                 f'dimensional')

        for i, atom in enumerate(self.atoms):
            atom.coord = value[i]

    def _check_idxs_are_present(self, *args):
        """Are a set of indexes present in the collection of atoms?

        Arguments:
            args (int):

        Raises:
            (ValueError):
        """
        if not set(args).issubset(set(range(self.n_atoms))):
            raise ValueError(f'Cannot calculate the value between {args}. '
                             f'At least one atom not present')
        return None

    def distance(self,
                 i: int,
                 j: int):
        """Distance between two atoms (Å)

        Arguments:
            i (int):
            j (int):

        Returns:
            (float): Distance

        Raises:
            (ValueError):
        """
        self._check_idxs_are_present(i, j)

        value = np.linalg.norm(self.atoms[i].coord - self.atoms[j].coord)

        return Distance(value)

    def angle(self,
              i: int,
              j: int,
              k: int):
        """
        Angle between three atoms i-j-k

        Arguments:
            i (int): Atom index of the left hand side in the angle
            j (int):                ---     middle    ---
            k (int):                -- right hand side --

        Returns:
            (autode.values.Angle):

        Raises:
            (ValueError):
        """
        self._check_idxs_are_present(i, j, k)

        vec1 = self.atoms[i].coord - self.atoms[j].coord
        vec2 = self.atoms[k].coord - self.atoms[j].coord

        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if np.isclose(norms, 0.0):
            raise ValueError(f'Cannot calculate the angle {i}-{j}-{k} - at '
                             f'least one zero vector')

        value = np.arccos(np.dot(vec1, vec2) / norms)

        return Angle(value)

    def dihedral(self,
                 w: int,
                 x: int,
                 y: int,
                 z: int):
        """
        Dihedral angle between four atoms

        Arguments:
            w (int): Atom index of the first atom in the dihedral
            x (int):               -- second --
            y (int):               -- third  --
            z (int):               -- fourth --

        Returns:
            (autode.values.Angle):

        Raises:
            (ValueError):
        """
        self._check_idxs_are_present(w, x, y, z)

        vec_yx = self.atoms[x].coord - self.atoms[y].coord
        vec_zw = self.atoms[w].coord - self.atoms[z].coord
        vec_yz = self.atoms[z].coord - self.atoms[y].coord

        vec1, vec2 = np.cross(vec_yx, vec_yz), np.cross(-vec_yz, vec_zw)

        # Normalise and ensure no zero vectors, for which the dihedral is not
        # defined
        for vec in (vec1, vec2, vec_yz):
            norm = np.linalg.norm(vec)

            if np.isclose(norm, 0.0):
                raise ValueError(f'Cannot calculate the dihedral angle '
                                 f'{w}-{x}-{y}-{z} - one zero vector')

            vec /= norm

        value = -np.arctan2(np.dot(np.cross(vec1, vec_yz), vec2),
                            np.dot(vec1, vec2))

        return Angle(value)

    def __init__(self, atoms=None):
        """
        Collection of atoms, used as a a base class for a species etc

        Arguments:
            atoms (list(autode.atoms.Atom)):
        """
        self.atoms = atoms


class Atom:

    def __repr__(self):
        x, y, z = self.coord
        return f'[{self.label}, {x:.4f}, {y:.4f}, {z:.4f}]'

    def __str__(self):
        return self.__repr__()

    @property
    def atomic_number(self):
        """Atomic numbers are the position in the elements, plus one"""
        return elements.index(self.label) + 1

    @property
    def atomic_symbol(self):
        """A more interpretable alias for label"""
        return self.label

    @property
    def is_metal(self):
        """Is this atom a metal?"""
        return self.label in metals

    @property
    def group(self):
        """Group of the periodic table is this atom in. 0 if not found"""

        for group_idx in range(1, 18):

            if self.label in PeriodicTable.group(group_idx):
                return group_idx

        return 0

    @property
    def period(self):
        """Group of the periodic table is this atom in. 0 if not found"""

        for period_idx in range(1, 7):

            if self.label in PeriodicTable.period(period_idx):
                return period_idx

        return 0

    @property
    def tm_row(self):
        """
        Row of transition metals that this element is in

        Returns:
            (int | None):
        """
        for row in [1, 2, 3]:
            if self.label in PeriodicTable.transition_metals(row):
                return row

        return None

    def translate(self, vec: np.ndarray):
        """
        Translate this atom by a vector

         Arguments:
             vec (np.ndarray): Shape = (3,)

        """
        self.coord += vec
        return None

    def rotate(self,
               axis: np.ndarray,
               theta: float,
               origin: Union[np.ndarray, None] = None):
        """Rotate this atom theta radians around an axis given an origin

        Arguments:
            axis (np.ndarray): Axis to rotate in. shape = (3,)
            theta (float): Angle in radians (float)

        Keyword Arguments:
            origin (np.ndarray): Rotate about this origin. shape = (3,)
                                 if no origin is specified then the atom
                                 is rotated without translation.
        """
        # If specified shift so that the origin is at (0, 0, 0)
        if origin is not None:
            self.translate(vec=-origin)

        # apply the rotation
        rot_matrix = get_rot_mat_euler(axis=axis, theta=theta)
        self.coord = np.matmul(rot_matrix, self.coord)

        # and shift back, if required
        if origin is not None:
            self.translate(vec=origin)

        return None

    def __init__(self, atomic_symbol, x=0.0, y=0.0, z=0.0):
        """
        Atom class. Centered at the origin by default

        Arguments:
            atomic_symbol (str): Symbol of an element e.g. 'C' for carbon

        Keyword Arguments:
            x (float): x coordinate in 3D space (Å)
            y (float): y coordinate in 3D space (Å)
            z (float): z coordinate in 3D space (Å)
        """
        assert atomic_symbol in elements

        self.label = atomic_symbol
        self.coord = np.array([float(x), float(y), float(z)])


class DummyAtom(Atom):

    @property
    def atomic_number(self):
        """The atomic number is defined as 0 for a dummy atom"""
        return 0

    def __init__(self, x, y, z):
        super().__init__('H', x, y, z)

        self.label = 'D'


elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
            'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
            'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
            'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
            'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


class PeriodicTable:

    table = np.array([['H',    '',   '',   '',   '',  '',    '',   '',   '',   '',   '',   '',   '',   '',   '',   '',   '', 'He'],
                      ['Li', 'Be',   '',   '',   '',  '',    '',   '',   '',   '',   '',   '',  'B',  'C',  'N',  'O',  'F', 'Ne'],
                      ['Na', 'Mg',   '',   '',   '',  '',    '',   '',   '',   '',   '',   '', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar'],
                      ['K',  'Ca', 'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
                      ['Rb', 'Sr',  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',  'I', 'Xe'],
                      ['Cs', 'Ba',   '', 'Hf', 'Ta',  'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
                      ['Fr', 'Ra',   '', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']],
                     dtype=str)

    @classmethod
    def period(cls, n: int):
        """Period of the periodic table, with 1 being the first period

        Arguments:
            n (int):

        Returns:
            (np.ndarray(str)):

        Raises:
            (ValueError): If n is not valid period index
        """
        if n < 1 or n > 7:
            raise ValueError('Not a valid period. Must be 1-7')

        # Exclude the empty strings of non-present elements
        return np.array([elem for elem in cls.table[n - 1, :] if elem != ''])

    @classmethod
    def group(cls, n: int):
        """Group of the periodic table, with 1 being the first period

        Arguments:
            n (int):

        Returns:
            (np.ndarray(str)):

        Raises:
            (ValueError): If n is not valid group index
        """
        if n < 1 or n > 18:
            raise ValueError('Not a valid group. Must be 1-18')

        # Exclude the empty strings of non-present elements
        return np.array([elem for elem in cls.table[:, n - 1] if elem != ''])

    @classmethod
    def element(cls, period: int, group: int):
        """
        Element given it's index in the periodic table, excluding
        lanthanides and actinides.

        Arguments:
            period (int):

            group (int):

        Returns:
            (str): Atomic symbol of the element

        Raises:
            (IndexError): If such an element does not exist
        """
        try:
            elem = cls.table[period-1, group-1]  # Convert from 1 -> 0 indexing
            assert elem != ''

        except (IndexError, AssertionError):
            raise IndexError('Index of the element not found')

        return elem

    @classmethod
    def transition_metals(cls, row: int):
        """
        Collection of transition metals (TMs) of a defined row. e.g.

        row = 1 -> [Sc, Ti .. Zn]

        Arguments:
            row (int): Colloquial name for TMs period

        Returns:
            (np.ndarray(str)):

        Raises:
            (ValueError): If the row is not valid
        """
        if row < 1 or row > 3:
            raise ValueError('Not a valid row of TMs. Must be 1-3')

        tms = [elem for elem in cls.period(row+3) if elem in metals]
        return np.array(tms, dtype=str)

    lanthanoids = lanthanides = np.array(['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'], dtype=str)
    actinoids = actinides = np.array(['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'], dtype=str)


# A set of reasonable valances for anionic/neutral/cationic atoms
valid_valances = {'H': [0, 1],
                  'B': [3, 4],
                  'C': [2, 3, 4],
                  'N': [2, 3, 4],
                  'O': [1, 2, 3],
                  'F': [0, 1],
                  'Si': [2, 3, 4],
                  'P': [2, 3, 4, 5, 6],
                  'S': [2, 3, 4, 5, 6],
                  'Cl': [0, 1, 2, 3, 4],
                  'Br': [0, 1, 2, 3, 4],
                  'I': [0, 1, 2, 3, 4, 5, 6],
                  'Rh': [0, 1, 2, 3, 4, 5, 6]
                  }

# masses from https://ciaaw.org/atomic-masses.htm
atomic_weights = {'H': 1.01, 'He': 4.0, 'Li': 6.94, 'Be': 9.01, 'B': 10.81, 'C': 12.01, 'N': 14.01, 'O': 16.0, 'F': 19.0, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.3,
                  'Al': 26.98, 'Si': 28.08, 'P': 30.97, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.95, 'K': 39.1, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.0,
                  'Mn': 54.94, 'Fe': 55.84, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.63, 'As': 74.92, 'Se': 78.97, 'Br': 79.9, 'Kr': 83.8,
                  'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.95, 'Tc': 97.91, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87,
                  'Cd': 112.41, 'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.9, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12,
                  'Pr': 140.91, 'Nd': 144.24, 'Pm': 144.91, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.5, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93,
                  'Yb': 173.04, 'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
                  'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.04, 'Pa': 231.04,
                  'U': 238.03, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 'Lr': 262.0,
                  'Rf': 267.0, 'Db': 268.0, 'Sg': 271.0, 'Bh': 274.0, 'Hs': 269.0, 'Mt': 276.0, 'Ds': 281.0, 'Rg': 281.0, 'Cn': 285.0, 'Nh': 286.0, 'Fl': 289.0, 'Mc': 288.0,
                  'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0}

# vdw radii from https://books.google.no/books?id=bNDMBQAAQBAJ
vdw_radii = {'H': 1.1, 'He': 1.4, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 'Al': 1.84,
             'Si': 2.1, 'P': 1.8, 'S': 1.8, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31, 'Sc': 2.15, 'Ti': 2.11, 'V': 2.07, 'Cr': 2.06, 'Mn': 2.05, 'Fe': 2.04,
             'Co': 2.0, 'Ni': 1.97, 'Cu': 1.96, 'Zn': 2.01, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.9, 'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.32,
             'Zr': 2.23, 'Nb': 2.18, 'Mo': 2.17, 'Tc': 2.16, 'Ru': 2.13, 'Rh': 2.1, 'Pd': 2.1, 'Ag': 2.11, 'Cd': 2.18, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06,
             'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68, 'La': 2.43, 'Ce': 2.42, 'Pr': 2.4, 'Nd': 2.39, 'Pm': 2.38, 'Sm': 2.36, 'Eu': 2.35, 'Gd': 2.34, 'Tb': 2.33,
             'Dy': 2.31, 'Ho': 2.3, 'Er': 2.29, 'Tm': 2.27, 'Yb': 2.26, 'Lu': 2.24, 'Hf': 2.23, 'Ta': 2.22, 'W': 2.18, 'Re': 2.16, 'Os': 2.16, 'Ir': 2.13, 'Pt': 2.13,
             'Au': 2.14, 'Hg': 2.23, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.2, 'Fr': 3.48, 'Ra': 2.83, 'Ac': 2.47, 'Th': 2.45, 'Pa': 2.43,
             'U': 2.41, 'Np': 2.39, 'Pu': 2.43, 'Am': 2.44, 'Cm': 2.45, 'Bk': 2.44, 'Cf': 2.45, 'Es': 2.45, 'Fm': 2.45, 'Md': 2.46, 'No': 2.46, 'Lr': 2.46}

pi_valencies = {'B': [1, 2], 'N': [1, 2], 'O': [1], 'C': [1, 2, 3], 'P': [1, 2, 3, 4], 'S': [1, 3, 4, 5],
                'Si': [1, 2, 3]}

metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
          'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce',
          'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
          'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
          'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl',
          'Mc', 'Lv']


def get_maximal_valance(atom_label):
    """Get the maximum valance of an atom

    Arguments:
        atom_label (str): atom label e.g. C or Pd

    Returns:
        (int): maximal valence of the atom
    """

    if atom_label in valid_valances.keys():
        return valid_valances[atom_label][-1]
    else:
        logger.warning(f'Could not find a valid valance for {atom_label}. '
                       f'Guessing at 6')
        return 6


def get_atomic_weight(atom_label):
    """Get the atomic weight of an atom

    Arguments:
        atom_label (str): atom label e.g. C or Pd

    Returns:
        (float): atomic weight of the atom
    """

    if atom_label in atomic_weights.keys():
        return atomic_weights[atom_label]
    else:
        logger.warning(f'Could not find a valid weight for {atom_label}. '
                       f'Guessing at 70')
        return 70


def get_vdw_radius(atom_label):
    """Get the van der waal's radius of an atom

    Arguments:
        atom_label (str): atom label e.g. C or Pd

    Returns:
        (float): van der waal's radius of the atom
    """
    if atom_label in vdw_radii.keys():
        return vdw_radii[atom_label]
    else:
        logger.error(f'Couldn\'t find the VdV radii for {atom_label}. '
                     f'Guessing at 2.3')
        return 2.3


def is_pi_atom(atom_label, valency):
    """
    Determine if an atom is a 'π-atom' i.e. is unsaturated and is a first or
    second row element

    Arguments:
        atom_label (str):
        valency (int):

    Returns:
        (bool)
    """

    if atom_label not in pi_valencies.keys():
        return False

    if valency in pi_valencies[atom_label]:
        return True

    return False
