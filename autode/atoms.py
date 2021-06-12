import numpy as np
from typing import Union, Optional, List
from autode.log import logger
from autode.geom import get_rot_mat_euler
from autode.values import (Distance, Angle, Mass, Coordinate,
                           Coordinates, MomentOfInertia)


class Atom:

    def __repr__(self):
        x, y, z = self.coord
        return f'Atom({self.label}, {x:.4f}, {y:.4f}, {z:.4f})'

    def __str__(self):
        return self.__repr__()

    @property
    def atomic_number(self) -> int:
        """Atomic numbers are the position in the elements, plus one"""
        return elements.index(self.label) + 1

    @property
    def atomic_symbol(self) -> str:
        """A more interpretable alias for label"""
        return self.label

    @property
    def coord(self) -> Coordinate:
        """Position of this atom in space

        Returns:
            (autode.values.Coordinate):
        """
        return self._coord

    @coord.setter
    def coord(self, *args):
        """
        Coordinate setter

        Arguments:
            *args (float | list(float) | np.ndarray(float)):

        Raises:
            (ValueError): If the arguments
        """
        self._coord = Coordinate(*args)

    @property
    def is_metal(self) -> bool:
        """Is this atom a metal?"""
        return self.label in metals

    @property
    def group(self) -> int:
        """Group of the periodic table is this atom in. 0 if not found"""

        for group_idx in range(1, 18):

            if self.label in PeriodicTable.group(group_idx):
                return group_idx

        return 0

    @property
    def period(self) -> int:
        """Group of the periodic table is this atom in. 0 if not found"""

        for period_idx in range(1, 7):

            if self.label in PeriodicTable.period(period_idx):
                return period_idx

        return 0

    @property
    def tm_row(self) -> Union[int, None]:
        """
        Row of transition metals that this element is in

        Returns:
            (int | None):
        """
        for row in [1, 2, 3]:
            if self.label in PeriodicTable.transition_metals(row):
                return row

        return None

    @property
    def weight(self) -> Mass:
        """
        Atomic weight

        Returns:
            (autode.values.Mass):
        """

        try:
            return Mass(atomic_weights[self.label])

        except KeyError:
            logger.warning(f'Could not find a valid weight for {self.label}. '
                           f'Guessing at 70')
            return Mass(70)

    def translate(self, vec: np.ndarray) -> None:
        """
        Translate this atom by a vector

        Arguments:
             vec (np.ndarray): Shape = (3,)
        """
        self.coord += vec
        return None

    def rotate(self,
               axis:   np.ndarray,
               theta:  float,
               origin: Union[np.ndarray, None] = None) -> None:
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

    # Method aliases
    mass = weight
    coordinate = coord

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
        self._coord = Coordinate(float(x), float(y), float(z))


class DummyAtom(Atom):

    @property
    def atomic_number(self):
        """The atomic number is defined as 0 for a dummy atom"""
        return 0

    @property
    def weight(self) -> Mass:
        """Dummy atoms do not have any weight/mass"""
        return Mass(0.0)

    def __init__(self, x, y, z):
        super().__init__('H', x, y, z)

        self.label = 'D'


class Atoms(list):

    def __repr__(self):
        return f'Atoms({super().__repr__()})'

    @property
    def coordinates(self) -> Coordinates:
        return Coordinates(np.array([a.coord for a in self]))

    @property
    def com(self) -> Coordinate:
        """
        Centre of mass of these coordinates

        COM = 1/M Σ_i m_i X_i

        where M is the total mass, m_i the mass of atom i and X_i it's
        coordinate

        Returns:
            (autode.values.Coordinate):
        """
        if len(self) == 0:
            raise ValueError('Undefined centre of mass with no atoms')

        com = Coordinate(0.0, 0.0, 0.0)

        for atom in self:
            com += atom.mass * atom.coord

        return Coordinate(com / sum(atom.mass for atom in self))

    @property
    def moi(self) -> MomentOfInertia:
        """
        Moment of inertia matrix (I):

            (I_00   I_01   I_02)
        I = (I_10   I_11   I_12)
            (I_20   I_21   I_22)

        Returns:
            (autode.values.MomentOfInertia):
        """
        moi = MomentOfInertia(np.zeros(shape=(3, 3)), units='amu Å^2')

        for atom in self:

            mass, (x, y, z) = atom.mass, atom.coord

            moi[0, 0] += mass * (y ** 2 + z ** 2)
            moi[0, 1] -= mass * (x * y)
            moi[0, 2] -= mass * (x * z)

            moi[1, 0] -= mass * (y * x)
            moi[1, 1] += mass * (x ** 2 + z ** 2)
            moi[1, 2] -= mass * (y * z)

            moi[2, 0] -= mass * (z * x)
            moi[2, 1] -= mass * (z * y)
            moi[2, 2] += mass * (x ** 2 + y ** 2)

        return moi

    def vector(self,
               i: int,
               j: int) -> np.ndarray:
        """
        Vector from atom i to atom j

        Arguments:
            i (int):
            j (int):

        Returns:
            (np.ndarray):

        Raises:
            (IndexError): If i or j are not present
        """
        return self[j].coord - self[i].coord

    def nvector(self,
               i: int,
               j: int) -> np.ndarray:
        """
        Normalised vector from atom i to atom j

        Arguments:
            i (int):
            j (int):

        Returns:
            (np.ndarray):

        Raises:
            (IndexError): If i or j are not present
        """
        vec = self.vector(i, j)
        return vec / np.linalg.norm(vec)

    def are_linear(self,
                   angle_tol: Angle = Angle(1, units='deg')) -> bool:
        """
        Are these set of atoms colinear?

        Arguments:
            angle_tol (autode.values.Angle): Tolerance on the angle

        Returns:
            (bool):
        """
        if len(self) < 2:      # Must have at least 2 atoms colinear
            return False

        if len(self) == 2:     # Two atoms must be linear
            return True

        tol = np.abs(1.0 - np.cos(angle_tol.to('rad')))

        # Take the normalised first vector
        vec0 = self[1].coord - self[0].coord
        vec0 /= np.linalg.norm(vec0)

        for atom in self[2:]:
            vec = atom.coord - self[0].coord
            cos_theta = np.dot(vec, vec0) / np.linalg.norm(vec)

            # Both e.g. <179° and >1° should satisfy this condition for
            # angle_tol = 1°
            if np.abs(np.abs(cos_theta) - 1) > tol:
                return False

        return True


class AtomCollection:

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this collection"""
        return 0 if self.atoms is None else len(self.atoms)

    @property
    def coordinates(self) -> Optional[Coordinates]:
        """Numpy array of coordinates"""
        if self.atoms is None:
            return None

        return self.atoms.coordinates

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
            atom.coord = Coordinate(*value[i])

    @property
    def atoms(self) -> Optional[Atoms]:
        """Constituent atoms of this collection"""
        return self._atoms

    @atoms.setter
    def atoms(self,
              value: Union[List[Atom], Atoms, None]):
        """Set the constituent atoms of this collection"""
        self._atoms = Atoms(value) if value is not None else None

    @property
    def com(self) -> Optional[Coordinate]:
        """Centre of mass of this atom collection

        Returns:
            (autode.values.Coordinate): COM

        Raises:
            (ValueError): If there are no atoms
        """
        return None if self.atoms is None else self.atoms.com

    @property
    def moi(self) -> Optional[MomentOfInertia]:
        """
        Moment of inertia matrix (I)

        Returns:
            (autode.values.MomentOfInertia):
        """
        return None if self.atoms is None else self.atoms.moi

    @property
    def weight(self) -> Mass:
        """
        Molecular weight

        Returns:
            (autode.values.Mass):
        """
        return sum(atom.mass for atom in self.atoms)

    def _idxs_are_present(self, *args):
        """Are a set of indexes present in the collection of atoms?

        Arguments:
            args (int):

        Returns:
            (bool):
        """
        return set(args).issubset(set(range(self.n_atoms)))

    def distance(self,
                 i: int,
                 j: int) -> Distance:
        """Distance between two atoms (Å), indexed from 0.

        Arguments:
            i (int): Atom index of the first atom
            j (int): Atom index of the second atom

        Returns:
            (autode.values.Distance): Distance

        Raises:
            (ValueError):
        """
        if not self._idxs_are_present(i, j):
            raise ValueError(f'Cannot calculate the distance between {i}-{j}. '
                             f'At least one atom not present')

        value = np.linalg.norm(self.atoms[i].coord - self.atoms[j].coord)

        return Distance(value)

    def angle(self,
              i: int,
              j: int,
              k: int) -> Angle:
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
        if not self._idxs_are_present(i, j, k):
            raise ValueError(f'Cannot calculate the angle between {i}-{j}-{k}.'
                             f' At least one atom not present')

        vec1 = self.atoms[i].coord - self.atoms[j].coord
        vec2 = self.atoms[k].coord - self.atoms[j].coord

        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if np.isclose(norms, 0.0):
            raise ValueError(f'Cannot calculate the angle {i}-{j}-{k} - at '
                             f'least one zero vector')

        value = np.arccos(np.dot(vec1, vec2) / norms)

        return Angle(value)

    def dihedral(self,
                 x: int,
                 y: int,
                 z: int,
                 w: int) -> Angle:
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
        if not self._idxs_are_present(w, x, y, z):
            raise ValueError(f'Cannot calculate the dihedral angle involving '
                             f'atoms {w}-{x}-{y}-{z}. At least one atom not '
                             f'present')

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

        """
        Dihedral angles are defined as from the IUPAC gold book: "the torsion 
        angle between groups A and D is then considered to be positive if 
        the bond A-B is rotated in a clockwise direction through less than
        180 degrees"
        """
        value = -np.arctan2(np.dot(np.cross(vec1, vec_yz), vec2),
                            np.dot(vec1, vec2))

        return Angle(value)

    # Method aliases
    centre_of_mass = com
    moment_of_inertia = moi
    mass = weight

    def __init__(self,
                 atoms: Union[List[Atom], Atoms, None] = None):
        """
        Collection of atoms, used as a a base class for a species etc

        Arguments:
            atoms (autode.atoms.Atoms | list(autode.atoms.Atom) | None):
        """
        self._atoms = Atoms(atoms) if atoms is not None else None


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


#  Atomic weights in amu from:
#  IUPAC-CIAWW's Atomic weights of the elements: Review 2000
atomic_weights = {"H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182,
                  "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994,
                  "F": 18.9984032, "Ne": 2.01797, "Na": 22.989770,
                  "Mg": 24.3050, "Al": 26.981538, "Si": 28.0855,
                  "P": 30.973761, "S": 32.065, "Cl": 35.453, "Ar": 39.948,
                  "K": 39.0983, "Ca": 40.078, "Sc": 44.955910, "Ti": 47.867,
                  "V": 50.9415, "Cr": 51.9961, "Mn": 54.938049,
                  "Fe": 55.845, "Co": 58.933200, "Ni": 58.6934,
                  "Cu": 63.546, "Zn": 65.409, "Ga": 69.723, "Ge": 72.64,
                  "As": 74.92160, "Se": 78.96, "Br": 79.904, "Kr": 83.798,
                  "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585, "Zr": 91.224,
                  "Nb": 92.90638, "Mo": 95.94, "Ru": 101.07,
                  "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682,
                  "Cd": 112.411, "In": 114.818, "Sn": 118.710,
                  "Sb": 121.760, "Te": 127.60, "I": 126.90447,
                  "Xe": 131.293, "Cs": 132.90545, "Ba": 137.327,
                  "La": 138.9055, "Ce": 140.116, "Pr": 140.90765,
                  "Nd": 144.24, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25,
                  "Tb": 158.92534, "Dy": 162.500, "Ho": 164.93032,
                  "Er": 167.259, "Tm": 168.93421, "Yb": 173.04,
                  "Lu": 174.967, "Hf": 178.49, "Ta": 180.9479,
                  "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
                  "Pt": 195.078, "Au": 196.96655, "Hg": 200.59,
                  "Tl": 204.3833, "Pb": 207.2, "Bi": 208.98038,
                  "Th": 232.0381, "Pa": 231.03588, "U": 238.02891,
                  # Remainder from https://ciaaw.org/atomic-masses.htm
                  'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0,
                  'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0,
                  'Md': 258.0, 'No': 259.0, 'Lr': 262.0, 'Rf': 267.0,
                  'Db': 268.0, 'Sg': 271.0, 'Bh': 274.0, 'Hs': 269.0,
                  'Mt': 276.0, 'Ds': 281.0, 'Rg': 281.0, 'Cn': 285.0,
                  'Nh': 286.0, 'Fl': 289.0, 'Mc': 288.0, 'Lv': 293.0,
                  'Ts': 294.0, 'Og': 294.0
                  }


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


# TODO ------ remove
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
