import numpy as np
from copy import deepcopy
from typing import Union, Optional, List, Sequence, Any
from autode.log import logger
from autode.geom import get_rot_mat_euler
from autode.values import (
    Distance,
    Angle,
    Mass,
    Coordinate,
    Coordinates,
    MomentOfInertia,
)


class Atom:
    def __init__(
        self,
        atomic_symbol: str,
        x: Any = 0.0,
        y: Any = 0.0,
        z: Any = 0.0,
        atom_class: Optional[int] = None,
        partial_charge: Optional[float] = None,
    ):
        """
        Atom class. Centered at the origin by default. Can be initialised from
        positional or keyword arguments:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('H')
            Atom(H, 0.0000, 0.0000, 0.0000)
            >>>
            >>> ade.Atom('H', x=1.0, y=1.0, z=1.0)
            Atom(H, 1.0000, 1.0000, 1.0000)
            >>>
            >>> ade.Atom('H', 1.0, 1.0, 1.0)
            Atom(H, 1.0000, 1.0000, 1.0000)

        -----------------------------------------------------------------------
        Arguments:
            atomic_symbol: Symbol of an element e.g. 'C' for carbon

            x: x coordinate in 3D space (Å)

            y: y coordinate in 3D space (Å)

            z: z coordinate in 3D space (Å)

            atom_class: Fictitious additional labels to distinguish otherwise
                        identical atoms. Useful in finding bond isomorphisms
                        over identity reactions

            partial_charge: Partial atomic charge in units of e, determined by
                            the atomic envrionment. Not an observable property.
        """
        assert atomic_symbol in elements

        self.label = atomic_symbol
        self._coord = Coordinate(float(x), float(y), float(z))
        self.atom_class = atom_class
        self.partial_charge = (
            None if partial_charge is None else float(partial_charge)
        )

    def __repr__(self):
        """
        Representation of this atom

        -----------------------------------------------------------------------
        Returns:
            (str): Representation
        """
        x, y, z = self.coord
        return f"Atom({self.label}, {x:.4f}, {y:.4f}, {z:.4f})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other: Any):
        """Equality of another atom to this one"""
        are_equal = (
            isinstance(other, Atom)
            and other.label == self.label
            and other.atom_class == self.atom_class
            and isinstance(other.partial_charge, type(self.partial_charge))
            and (
                (other.partial_charge is None and self.partial_charge is None)
                or np.isclose(other.partial_charge, self.partial_charge)
            )
            and np.allclose(other._coord, self._coord)
        )
        return are_equal

    @property
    def atomic_number(self) -> int:
        """
        Atomic numbers are the position in the elements (indexed from zero),
        plus one. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> atom = ade.Atom('C')
            >>> atom.atomic_number
            6

        -----------------------------------------------------------------------
        Returns:
            (int): Atomic number
        """
        return elements.index(self.label) + 1

    @property
    def atomic_symbol(self) -> str:
        """
        A more interpretable alias for Atom.label. Should be present in the
        elements. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> atom = ade.Atom('Zn')
            >>> atom.atomic_symbol
            'Zn'

        -----------------------------------------------------------------------
        Returns:
            (str): Atomic symbol
        """
        return self.label

    @property
    def coord(self) -> Coordinate:
        """
        Position of this atom in space. Coordinate has attributes x, y, z
        for the Cartesian displacements. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> atom = ade.Atom('H')
            >>> atom.coord
            Coordinate([0. 0. 0.] Å)

        To initialise at a different position away from the origin

        .. code-block:: Python

            >>> ade.Atom('H', x=1.0).coord
            Coordinate([1. 0. 0.] Å)
            >>> ade.Atom('H', x=1.0).coord.x
            1.0

        Coordinates are instances of autode.values.ValueArray, so can
        be converted from the default angstrom units to e.g. Bohr

        .. code-block:: Python

            >>> ade.Atom('H', x=1.0, y=-1.0).coord.to('a0')
            Coordinate([1.889  -1.889  0. ] bohr)

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Coordinate): Coordinate
        """
        return self._coord

    @coord.setter
    def coord(self, *args):
        """
        Coordinate setter

        -----------------------------------------------------------------------
        Arguments:
            *args (float | list(float) | np.ndarray(float)):

        Raises:
            (ValueError): If the arguments cannot be coerced into a (3,) shape
        """
        self._coord = Coordinate(*args)

    @property
    def is_metal(self) -> bool:
        """
        Is this atom a metal? Defines metals to be up to and including:
        Ga, Sn, Bi. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('C').is_metal
            False
            >>> ade.Atom('Zn').is_metal
            True

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return self.label in metals

    @property
    def group(self) -> int:
        """
        Group of the periodic table is this atom in. 0 if not found. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('C').group
            14

        -----------------------------------------------------------------------
        Returns:
            (int): Group
        """

        for group_idx in range(1, 18):
            if self.label in PeriodicTable.group(group_idx):
                return group_idx

        return 0

    @property
    def period(self) -> int:
        """
        Period of the periodic table is this atom in. 0 if not found. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('C').period
            2

        -----------------------------------------------------------------------
        Returns:
            (int): Period
        """

        for period_idx in range(1, 7):
            if self.label in PeriodicTable.period(period_idx):
                return period_idx

        return 0

    @property
    def tm_row(self) -> Optional[int]:
        """
        Row of transition metals that this element is in. Returns None if
        this atom is not a metal. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('Zn').tm_row
            1

        -----------------------------------------------------------------------
        Returns:
            (int | None): Transition metal row
        """
        for row in [1, 2, 3]:
            if self.label in PeriodicTable.transition_metals(row):
                return row

        return None

    @property
    def weight(self) -> Mass:
        """
        Atomic weight. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('C').weight
            Mass(12.0107 amu)
            >>>
            >>> ade.Atom('C').weight == ade.Atom('C').mass
            True

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Mass): Weight
        """

        try:
            return Mass(atomic_weights[self.label])

        except KeyError:
            logger.warning(
                f"Could not find a valid weight for {self.label}. "
                f"Guessing at 70"
            )
            return Mass(70)

    @property
    def mass(self) -> Mass:
        """Alias of weight. Returns Atom.weight an so can be converted
        to different units. For example, to convert the mass to electron masses:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('H').mass.to('me')
            Mass(1837.36222 m_e)

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Mass): Mass
        """
        return self.weight

    @property
    def maximal_valance(self) -> int:
        """
        The maximum/maximal valance that this atom supports in any charge
        state (most commonly). i.e. for H the maximal_valance=1. Useful for
        generating molecular graphs

        -----------------------------------------------------------------------
        Returns:
            (int): Maximal valance
        """

        if self.is_metal:
            return 6

        if self.label in _max_valances:
            return _max_valances[self.label]

        logger.warning(
            f"Could not find a valid valance for {self}. " f"Guessing at 6"
        )
        return 6

    @property
    def vdw_radius(self) -> Distance:
        """
        Van der Waals radius for this atom. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('H').vdw_radius
            Distance(1.1 Å)

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Distance): Van der Waals radius
        """

        if self.label in vdw_radii:
            radius = vdw_radii[self.label]
        else:
            logger.error(
                f"Couldn't find the VdV radii for {self}. "
                f"Guessing at 2.3 Å"
            )
            radius = 2.3

        return Distance(radius, "Å")

    @property
    def covalent_radius(self) -> Distance:
        """
        Covalent radius for this atom. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('H').covalent_radius
            Distance(0.31 Å)

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Distance): Van der Waals radius
        """
        radius = Distance(
            _covalent_radii_pm[self.atomic_number - 1], units="pm"
        )
        return radius.to("Å")

    def is_pi(self, valency: int) -> bool:
        """
        Determine if this atom is a 'π-atom' i.e. is unsaturated. Only
        approximate! Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> ade.Atom('C').is_pi(valency=3)
            True
            >>> ade.Atom('H').is_pi(valency=1)
            False

        -----------------------------------------------------------------------
        Arguments:
            valency (int):

        Returns:
            (bool):
        """

        if self.label in non_pi_elements:
            return False

        if self.label not in pi_valencies:
            logger.warning(
                f"{self.label} not found in π valency dictionary - "
                f"assuming not a π-atom"
            )
            return False

        if valency in pi_valencies[self.label]:
            return True

        return False

    def translate(self, *args, **kwargs) -> None:
        """
        Translate this atom by a vector in place. Arguments should be
        coercible into a coordinate (i.e. length 3). Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> atom = ade.Atom('H')
            >>> atom.translate(1.0, 0.0, 0.0)
            >>> atom.coord
            Coordinate([1. 0. 0.] Å)

        Atoms can also be translated using numpy arrays:

        .. code-block:: Python

            >>> import autode as ade
            >>> import numpy as np
            >>>
            >>> atom = ade.Atom('H')
            >>> atom.translate(np.ones(3))
            >>> atom.coord
            Coordinate([1. 1. 1.] Å)
            >>>
            >>> atom.translate(vec=-atom.coord)
            >>> atom.coord
            Coordinate([0. 0. 0.] Å)

        -----------------------------------------------------------------------
        Arguments:
             *args (float | np.ndarray | list(float)):

        Keyword Arguments:
            vec (np.ndarray): Shape = (3,)
        """
        if "vec" in kwargs:
            # Assume the vec is cast-able to a numpy array which can be added
            self.coord += np.asarray(kwargs["vec"])

        elif len(kwargs) > 0:
            raise ValueError(
                f"Expecting only a vec keyword argument. " f"Had {kwargs}"
            )

        else:
            self.coord += Coordinate(*args)

        return None

    def rotate(
        self,
        axis: Union[np.ndarray, Sequence],
        theta: Union[Angle, float],
        origin: Union[np.ndarray, Sequence, None] = None,
    ) -> None:
        """
        Rotate this atom theta radians around an axis given an origin. By
        default the rotation is applied around the origin with the angle
        in radians (unless an autode.values.Angle). Rotation is applied in
        place. To rotate a H atom around the z-axis:

        .. code-block:: Python

            >>> import autode as ade
            >>> atom = ade.Atom('H', x=1.0)
            >>> atom.rotate(axis=[0.0, 0.0, 1.0], theta=3.14)
            >>> atom.coord
            Coordinate([-1.  0.  0.] Å)

        With an origin:

        .. code-block:: Python

          >>> import autode as ade
          >>> atom = ade.Atom('H')
          >>> atom.rotate(axis=[0.0, 0.0, 1.0], theta=3.14, origin=[1.0, 0.0, 0.0])
          >>> atom.coord
          Coordinate([2.  0.  0.] Å)

        And with an angle not in radians:

        .. code-block:: Python

            >>> import autode as ade
            >>> from autode.values import Angle
            >>>
            >>> atom = ade.Atom('H', x=1.0)
            >>> atom.rotate(axis=[0.0, 0.0, 1.0], theta=Angle(180, units='deg'))
            >>> atom.coord
            Coordinate([-1.  0.  0.] Å)

        -----------------------------------------------------------------------
        Arguments:
            axis: Axis to rotate in. shape = (3,)

            theta: Angle to rotate by

            origin: Rotate about this origin. shape = (3,) if no origin is
                    specified then the atom is rotated without translation.
        """
        # If specified, shift so that the origin is at (0, 0, 0)
        if origin is not None:
            self.translate(vec=-np.asarray(origin))

        # apply the rotation
        rot_matrix = get_rot_mat_euler(axis=axis, theta=theta)
        self.coord = np.matmul(rot_matrix, self.coord)

        # and shift back, if required
        if origin is not None:
            self.translate(vec=np.asarray(origin))

        return None

    def copy(self) -> "Atom":
        return deepcopy(self)

    # --- Method aliases ---
    coordinate = coord


class DummyAtom(Atom):
    def __init__(self, x, y, z):
        """
        Dummy atom

        -----------------------------------------------------------------------
        Arguments:
            x (float): x coordinate in 3D space (Å)
            y (float): y
            z (float): z
        """
        # Superclass constructor called with a valid element...
        super().__init__("H", x, y, z)

        # then re-assigned
        self.label = "D"

    @property
    def atomic_number(self):
        """The atomic number is defined as 0 for a dummy atom"""
        return 0

    @property
    def weight(self) -> Mass:
        """Dummy atoms do not have any weight/mass"""
        return Mass(0.0)

    @property
    def mass(self) -> Mass:
        """Dummy atoms do not have any weight/mass"""
        return Mass(0.0)

    @property
    def vdw_radius(self) -> Distance:
        """Dummy atoms have no radius"""
        return Distance(0.0, units="Å")

    @property
    def covalent_radius(self) -> Distance:
        """Dummy atoms have no radius"""
        return Distance(0.0, units="Å")


class Atoms(list):
    def __repr__(self):
        """Representation"""
        return f"Atoms(n_atoms={len(self)}, {super().__repr__()})"

    def __add__(self, other):
        """Add another set of Atoms to this one. Can add None"""
        if other is None:
            return self

        return super().__add__(other)

    def __radd__(self, other):
        """Add another set of Atoms to this one. Can add None"""
        return self.__add__(other)

    def copy(self) -> "Atoms":
        """
        Copy these atoms, deeply

        -----------------------------------------------------------------------
        Returns:
             (autode.atoms.Atoms):
        """
        return deepcopy(self)

    def remove_dummy(self) -> None:
        """Remove all the dummy atoms from this list of atoms"""

        for i, atom in enumerate(self):
            if isinstance(atom, DummyAtom):
                del self[i]
        return

    @property
    def coordinates(self) -> Coordinates:
        return Coordinates(np.array([a.coord for a in self]))

    @coordinates.setter
    def coordinates(self, value: np.ndarray):
        """Set the coordinates from a numpy array

        -----------------------------------------------------------------------
        Arguments:
            value (np.ndarray): Shape = (n_atoms, 3) or (3*n_atoms) as a
                                row major vector
        """

        if value.ndim == 1:
            assert value.shape == (3 * len(self),)
            value = value.reshape((-1, 3))

        elif value.ndim == 2:
            assert value.shape == (len(self), 3)

        else:
            raise AssertionError(
                "Cannot set coordinates from a array with"
                f"shape: {value.shape}. Must be 1 or 2 "
                f"dimensional"
            )

        for i, atom in enumerate(self):
            atom.coord = Coordinate(*value[i])

    @property
    def com(self) -> Coordinate:
        r"""
        Centre of mass of these coordinates

        .. math::
            \text{COM} = \frac{1}{M} \sum_i m_i R_i

        where M is the total mass, m_i the mass of atom i and R_i it's
        coordinate

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Coordinate): COM
        """
        if len(self) == 0:
            raise ValueError("Undefined centre of mass with no atoms")

        com = Coordinate(0.0, 0.0, 0.0)

        for atom in self:
            com += atom.mass * atom.coord

        return Coordinate(com / sum(atom.mass for atom in self))

    @property
    def moi(self) -> MomentOfInertia:
        """
        Moment of inertia matrix (I)::

                (I_00   I_01   I_02)
            I = (I_10   I_11   I_12)
                (I_20   I_21   I_22)

        Returns:
            (autode.values.MomentOfInertia):
        """
        moi = MomentOfInertia(np.zeros(shape=(3, 3)), units="amu Å^2")

        for atom in self:
            mass, (x, y, z) = atom.mass, atom.coord

            moi[0, 0] += mass * (y**2 + z**2)
            moi[0, 1] -= mass * (x * y)
            moi[0, 2] -= mass * (x * z)

            moi[1, 0] -= mass * (y * x)
            moi[1, 1] += mass * (x**2 + z**2)
            moi[1, 2] -= mass * (y * z)

            moi[2, 0] -= mass * (z * x)
            moi[2, 1] -= mass * (z * y)
            moi[2, 2] += mass * (x**2 + y**2)

        return moi

    @property
    def contain_metals(self) -> bool:
        """
        Do these atoms contain at least a single metal atom?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return any(atom.label in metals for atom in self)

    def idxs_are_present(self, *args: int) -> bool:
        """Are all these indexes present in this set of atoms"""
        return set(args).issubset(set(range(len(self))))

    def eqm_bond_distance(self, i: int, j: int) -> Distance:
        """
        Equilibrium distance between two atoms. If known then use the
        experimental dimer distance, otherwise estimate if from the
        covalent radii of the two atoms. Example

        Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> mol = ade.Molecule(atoms=[ade.Atom('H'), ade.Atom('H')])
            >>> mol.distance(0, 1)
            Distance(0.0 Å)
            >>> mol.eqm_bond_distance(0, 1)
            Distance(0.741 Å)

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Distance): Equlirbium distance
        """
        if not self.idxs_are_present(i, j):
            raise ValueError(
                f"Cannot calculate the equilibrium distance "
                f"between {i}-{j}. At least one atom not present"
            )

        if i == j:
            return Distance(0.0, units="Å")

        symbols = f"{self[i].atomic_symbol}{self[j].atomic_symbol}"

        if symbols in _bond_lengths:
            return Distance(_bond_lengths[symbols], units="Å")

        # TODO: Something more accurate here
        return self[i].covalent_radius + self[j].covalent_radius

    def distance(self, i: int, j: int) -> Distance:
        """
        Distance between two atoms (Å), indexed from 0.

        .. code-block:: Python

            >>> import autode as ade
            >>> mol = ade.Molecule(atoms=[ade.Atom('H'), ade.Atom('H', x=1.0)])
            >>> mol.distance(0, 1)
            Distance(1.0 Å)

        -----------------------------------------------------------------------
        Arguments:
            i (int): Atom index of the first atom
            j (int): Atom index of the second atom

        Returns:
            (autode.values.Distance): Distance

        Raises:
            (ValueError):
        """
        if not self.idxs_are_present(i, j):
            raise ValueError(
                f"Cannot calculate the distance between {i}-{j}. "
                f"At least one atom not present"
            )

        return Distance(np.linalg.norm(self[i].coord - self[j].coord))

    def vector(self, i: int, j: int) -> np.ndarray:
        """
        Vector from atom i to atom j

        -----------------------------------------------------------------------
        Arguments:
            i (int):
            j (int):

        Returns:
            (np.ndarray):

        Raises:
            (IndexError): If i or j are not present
        """
        return self[j].coord - self[i].coord

    def nvector(self, i: int, j: int) -> np.ndarray:
        """
        Normalised vector from atom i to atom j

        -----------------------------------------------------------------------
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

    def are_linear(self, angle_tol: Angle = Angle(1, "º")) -> bool:
        """
        Are these set of atoms colinear?

        -----------------------------------------------------------------------
        Arguments:
            angle_tol (autode.values.Angle): Tolerance on the angle

        Returns:
            (bool): Whether the atoms are linear
        """
        if len(self) < 2:  # Must have at least 2 atoms colinear
            return False

        if len(self) == 2:  # Two atoms must be linear
            return True

        tol = np.abs(1.0 - np.cos(angle_tol.to("rad")))

        vec0 = self.nvector(0, 1)  # Normalised first vector

        for atom in self[2:]:
            vec = atom.coord - self[0].coord
            cos_theta = np.dot(vec, vec0) / np.linalg.norm(vec)

            # Both e.g. <179° and >1° should satisfy this condition for
            # angle_tol = 1°
            if np.abs(np.abs(cos_theta) - 1) > tol:
                return False

        return True

    def are_planar(self, distance_tol: Distance = Distance(1e-3, "Å")) -> bool:
        """
        Do all the atoms in this set lie in a single plane?

        -----------------------------------------------------------------------
        Arguments:
            distance_tol (autode.values.Distance):

        Returns:
            (bool):
        """
        if len(self) < 4:  # 3 points must lie in a plane
            return True

        arr = self.coordinates.to("Å")

        if isinstance(distance_tol, Distance):
            distance_tol_float = float(distance_tol.to("Å"))

        else:
            logger.warning("Assuming a distance tolerance in units of Å")
            distance_tol_float = float(distance_tol)

        # Calculate a normal vector to the first two atomic vectors from atom 0
        x0 = arr[0, :]
        normal_vec = np.cross(arr[1, :] - x0, arr[2, :] - x0)

        for i in range(3, len(self)):
            # Calculate the 0->i atomic vector, which must not have any
            # component in the direction in the normal if the atoms are planar
            if np.dot(normal_vec, arr[i, :] - x0) > distance_tol_float:
                return False

        return True


class AtomCollection:
    def __init__(self, atoms: Union[List[Atom], Atoms, None] = None):
        """
        Collection of atoms, used as a base class for a species, complex
        or transition state.

        -----------------------------------------------------------------------
        Arguments:
            atoms (autode.atoms.Atoms | list(autode.atoms.Atom) | None):
        """
        self._atoms = Atoms(atoms) if atoms is not None else None

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
    def coordinates(self, value: np.ndarray):
        """Set the coordinates from a numpy array

        -----------------------------------------------------------------------
        Arguments:
            value (np.ndarray): Shape = (n_atoms, 3) or (3*n_atoms) as a
                                row major vector
        """
        if self._atoms is None:
            raise ValueError(
                "Must have atoms set to be able to set the "
                "coordinates of them"
            )

        self._atoms.coordinates = value

    @property
    def atoms(self) -> Optional[Atoms]:
        """Constituent atoms of this collection"""
        return self._atoms

    @atoms.setter
    def atoms(self, value: Union[List[Atom], Atoms, None]):
        """Set the constituent atoms of this collection"""
        self._atoms = Atoms(value) if value is not None else None

    @property
    def com(self) -> Optional[Coordinate]:
        """Centre of mass of this atom collection

        -----------------------------------------------------------------------
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

        -----------------------------------------------------------------------
        Returns:
            (autode.values.MomentOfInertia):
        """
        return None if self.atoms is None else self.atoms.moi

    @property
    def weight(self) -> Mass:
        """
        Molecular weight

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Mass):
        """
        if self.n_atoms == 0:
            return Mass(0.0)

        return sum(atom.mass for atom in self.atoms)  # type: ignore

    def distance(self, i: int, j: int) -> Distance:
        assert self.atoms is not None, "Must have atoms"
        return self.atoms.distance(i, j)

    def eqm_bond_distance(self, i: int, j: int) -> Distance:
        assert self.atoms is not None, "Must have atoms"
        return self.atoms.eqm_bond_distance(i, j)

    def angle(self, i: int, j: int, k: int) -> Angle:
        r"""
        Angle between three atoms i-j-k, where the atoms are indexed from
        zero::

            E_i  --- E_j
                        \
                  θ      E_k


        Example:

        .. code-block:: Python

            >>> from autode import Atom, Molecule
            >>> h2o = Molecule(atoms=[Atom('H', x=-1), Atom('O'), Atom('H', x=1)])
            >>> h2o.angle(0, 1, 2).to('deg')
            Angle(180.0 °)


        -----------------------------------------------------------------------
        Arguments:
            i (int): Atom index of the left hand side in the angle
            j (int):  --- middle
            k (int):  --- right

        Returns:
            (autode.values.Angle): Angle

        Raises:
            (ValueError): If any of the atom indexes are not present
        """
        assert self.atoms is not None, "Must have atoms"

        if not self.atoms.idxs_are_present(i, j, k):
            raise ValueError(
                f"Cannot calculate the angle between {i}-{j}-{k}."
                f" At least one atom not present"
            )

        vec1 = self.atoms[i].coord - self.atoms[j].coord
        vec2 = self.atoms[k].coord - self.atoms[j].coord

        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if np.isclose(norms, 0.0):
            raise ValueError(
                f"Cannot calculate the angle {i}-{j}-{k} - at "
                f"least one zero vector"
            )

        # Cos(theta) must lie within [-1, 1]
        cos_value = np.clip(np.dot(vec1, vec2) / norms, a_min=-1, a_max=1)

        return Angle(np.arccos(cos_value))

    def dihedral(self, w: int, x: int, y: int, z: int) -> Angle:
        r"""
        Dihedral angle between four atoms (x, y, z, w), where the atoms are
        indexed from zero::

            E_w  --- E_x
                        \  φ
                         \
                          E_y ---- E_z

        Example:

        .. code-block:: Python

            >>> from autode import Atom, Molecule
            >>> h2s2 = Molecule(atoms=[Atom('S',  0.1527, 0.9668, -0.9288),
            ...                        Atom('S',  2.0024, 0.0443, -0.4227),
            ...                        Atom('H', -0.5802, 0.0234, -0.1850),
            ...                        Atom('H',  2.1446, 0.8424,  0.7276)])
            >>> h2s2.dihedral(2, 0, 1, 3).to('deg')
            Angle(-90.0 °)

        -----------------------------------------------------------------------
        Arguments:
            w (int): Atom index of the first atom in the dihedral
            x (int):               -- second --
            y (int):               -- third  --
            z (int):               -- fourth --

        Returns:
            (autode.values.Angle): Dihedral angle

        Raises:
            (ValueError): If any of the atom indexes are not present in the
                          molecule
        """
        assert self.atoms is not None, "Must have atoms"

        if not self.atoms.idxs_are_present(w, x, y, z):
            raise ValueError(
                f"Cannot calculate the dihedral angle involving "
                f"atoms {z}-{w}-{x}-{y}. At least one atom not "
                f"present"
            )

        vec_xw = self.atoms[w].coord - self.atoms[x].coord
        vec_yz = self.atoms[z].coord - self.atoms[y].coord
        vec_xy = self.atoms[y].coord - self.atoms[x].coord

        vec1, vec2 = np.cross(vec_xw, vec_xy), np.cross(-vec_xy, vec_yz)

        # Normalise and ensure no zero vectors, for which the dihedral is not
        # defined
        for vec in (vec1, vec2, vec_xy):
            norm = np.linalg.norm(vec)

            if np.isclose(norm, 0.0):
                raise ValueError(
                    f"Cannot calculate the dihedral angle "
                    f"{z}-{w}-{x}-{y} - one zero vector"
                )
            vec /= norm

        """
        Dihedral angles are defined as from the IUPAC gold book: "the torsion
        angle between groups A and D is then considered to be positive if
        the bond A-B is rotated in a clockwise direction through less than
        180 degrees"
        """
        value = -np.arctan2(
            np.dot(np.cross(vec1, vec_xy), vec2), np.dot(vec1, vec2)
        )

        return Angle(value)

    # --- Method aliases ---
    centre_of_mass = com
    moment_of_inertia = moi
    mass = weight


elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


class PeriodicTable:
    # fmt: off
    table = np.array(
        [['H',    '',   '',   '',   '',  '',    '',   '',   '',   '',   '',   '',   '',   '',   '',   '',   '', 'He'],
         ['Li', 'Be',   '',   '',   '',  '',    '',   '',   '',   '',   '',   '',  'B',  'C',  'N',  'O',  'F', 'Ne'],
         ['Na', 'Mg',   '',   '',   '',  '',    '',   '',   '',   '',   '',   '', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar'],
         ['K',  'Ca', 'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
         ['Rb', 'Sr',  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',  'I', 'Xe'],
         ['Cs', 'Ba',   '', 'Hf', 'Ta',  'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
         ['Fr', 'Ra',   '', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']],
        dtype=str
    )
    # fmt: on

    @classmethod
    def period(cls, n: int):
        """
        Period of the periodic table, with 1 being the first period

        -----------------------------------------------------------------------
        Arguments:
            n (int):

        Returns:
            (np.ndarray(str)):

        Raises:
            (ValueError): If n is not valid period index
        """
        if n < 1 or n > 7:
            raise ValueError("Not a valid period. Must be 1-7")

        # Exclude the empty strings of non-present elements
        return np.array([elem for elem in cls.table[n - 1, :] if elem != ""])

    @classmethod
    def group(cls, n: int):
        """
        Group of the periodic table, with 1 being the first period

        -----------------------------------------------------------------------
        Arguments:
            n (int):

        Returns:
            (np.ndarray(str)):

        Raises:
            (ValueError): If n is not valid group index
        """
        if n < 1 or n > 18:
            raise ValueError("Not a valid group. Must be 1-18")

        # Exclude the empty strings of non-present elements
        return np.array([elem for elem in cls.table[:, n - 1] if elem != ""])

    @classmethod
    def element(cls, period: int, group: int):
        """
        Element given it's index in the periodic table, excluding
        lanthanides and actinides.

        -----------------------------------------------------------------------
        Arguments:
            period (int):

            group (int):

        Returns:
            (str): Atomic symbol of the element

        Raises:
            (IndexError): If such an element does not exist
        """
        try:
            elem = cls.table[
                period - 1, group - 1
            ]  # Convert from 1 -> 0 indexing
            assert elem != ""

        except (IndexError, AssertionError):
            raise IndexError("Index of the element not found")

        return elem

    @classmethod
    def transition_metals(cls, row: int):
        """
        Collection of transition metals (TMs) of a defined row. e.g.

        row = 1 -> [Sc, Ti .. Zn]

        -----------------------------------------------------------------------
        Arguments:
            row (int): Colloquial name for TMs period

        Returns:
            (np.ndarray(str)):

        Raises:
            (ValueError): If the row is not valid
        """
        if row < 1 or row > 3:
            raise ValueError("Not a valid row of TMs. Must be 1-3")

        tms = [elem for elem in cls.period(row + 3) if elem in metals]
        return np.array(tms, dtype=str)

    lanthanoids = lanthanides = np.array(
        [
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ],
        dtype=str,
    )
    actinoids = actinides = np.array(
        [
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
        ],
        dtype=str,
    )


# A set of reasonable valances for anionic/neutral/cationic atoms
valid_valances = {
    "H": [0, 1],
    "B": [3, 4],
    "C": [2, 3, 4],
    "N": [2, 3, 4],
    "O": [1, 2, 3],
    "F": [0, 1],
    "Si": [2, 3, 4],
    "P": [2, 3, 4, 5, 6],
    "S": [2, 3, 4, 5, 6],
    "Cl": [0, 1, 2, 3, 4],
    "Br": [0, 1, 2, 3, 4],
    "I": [0, 1, 2, 3, 4, 5, 6],
    "Rh": [0, 1, 2, 3, 4, 5, 6],
}

#  Atomic weights in amu from:
#  IUPAC-CIAWW's Atomic weights of the elements: Review 2000
atomic_weights = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 2.01797,
    "Na": 22.989770,
    "Mg": 24.3050,
    "Al": 26.981538,
    "Si": 28.0855,
    "P": 30.973761,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955910,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938049,
    "Fe": 55.845,
    "Co": 58.933200,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.409,
    "Ga": 69.723,
    "Ge": 72.64,
    "As": 74.92160,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90585,
    "Zr": 91.224,
    "Nb": 92.90638,
    "Mo": 95.94,
    "Ru": 101.07,
    "Rh": 102.90550,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545,
    "Ba": 137.327,
    "La": 138.9055,
    "Ce": 140.116,
    "Pr": 140.90765,
    "Nd": 144.24,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92534,
    "Dy": 162.500,
    "Ho": 164.93032,
    "Er": 167.259,
    "Tm": 168.93421,
    "Yb": 173.04,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.078,
    "Au": 196.96655,
    "Hg": 200.59,
    "Tl": 204.3833,
    "Pb": 207.2,
    "Bi": 208.98038,
    "Th": 232.0381,
    "Pa": 231.03588,
    "U": 238.02891,
    # Remainder from https://ciaaw.org/atomic-masses.htm
    "Np": 237.0,
    "Pu": 244.0,
    "Am": 243.0,
    "Cm": 247.0,
    "Bk": 247.0,
    "Cf": 251.0,
    "Es": 252.0,
    "Fm": 257.0,
    "Md": 258.0,
    "No": 259.0,
    "Lr": 262.0,
    "Rf": 267.0,
    "Db": 268.0,
    "Sg": 271.0,
    "Bh": 274.0,
    "Hs": 269.0,
    "Mt": 276.0,
    "Ds": 281.0,
    "Rg": 281.0,
    "Cn": 285.0,
    "Nh": 286.0,
    "Fl": 289.0,
    "Mc": 288.0,
    "Lv": 293.0,
    "Ts": 294.0,
    "Og": 294.0,
}

# van der Walls radii from https://books.google.no/books?id=bNDMBQAAQBAJ
vdw_radii = {
    "H": 1.1,
    "He": 1.4,
    "Li": 1.82,
    "Be": 1.53,
    "B": 1.92,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Al": 1.84,
    "Si": 2.1,
    "P": 1.8,
    "S": 1.8,
    "Cl": 1.75,
    "Ar": 1.88,
    "K": 2.75,
    "Ca": 2.31,
    "Sc": 2.15,
    "Ti": 2.11,
    "V": 2.07,
    "Cr": 2.06,
    "Mn": 2.05,
    "Fe": 2.04,
    "Co": 2.0,
    "Ni": 1.97,
    "Cu": 1.96,
    "Zn": 2.01,
    "Ga": 1.87,
    "Ge": 2.11,
    "As": 1.85,
    "Se": 1.9,
    "Br": 1.85,
    "Kr": 2.02,
    "Rb": 3.03,
    "Sr": 2.49,
    "Y": 2.32,
    "Zr": 2.23,
    "Nb": 2.18,
    "Mo": 2.17,
    "Tc": 2.16,
    "Ru": 2.13,
    "Rh": 2.1,
    "Pd": 2.1,
    "Ag": 2.11,
    "Cd": 2.18,
    "In": 1.93,
    "Sn": 2.17,
    "Sb": 2.06,
    "Te": 2.06,
    "I": 1.98,
    "Xe": 2.16,
    "Cs": 3.43,
    "Ba": 2.68,
    "La": 2.43,
    "Ce": 2.42,
    "Pr": 2.4,
    "Nd": 2.39,
    "Pm": 2.38,
    "Sm": 2.36,
    "Eu": 2.35,
    "Gd": 2.34,
    "Tb": 2.33,
    "Dy": 2.31,
    "Ho": 2.3,
    "Er": 2.29,
    "Tm": 2.27,
    "Yb": 2.26,
    "Lu": 2.24,
    "Hf": 2.23,
    "Ta": 2.22,
    "W": 2.18,
    "Re": 2.16,
    "Os": 2.16,
    "Ir": 2.13,
    "Pt": 2.13,
    "Au": 2.14,
    "Hg": 2.23,
    "Tl": 1.96,
    "Pb": 2.02,
    "Bi": 2.07,
    "Po": 1.97,
    "At": 2.02,
    "Rn": 2.2,
    "Fr": 3.48,
    "Ra": 2.83,
    "Ac": 2.47,
    "Th": 2.45,
    "Pa": 2.43,
    "U": 2.41,
    "Np": 2.39,
    "Pu": 2.43,
    "Am": 2.44,
    "Cm": 2.45,
    "Bk": 2.44,
    "Cf": 2.45,
    "Es": 2.45,
    "Fm": 2.45,
    "Md": 2.46,
    "No": 2.46,
    "Lr": 2.46,
}

"""
Although a π-bond may not be well defined, it is useful to have a notion of
a bond about which there is restricted rotation. The below sets are used to
define which atoms may be π-bonded to another
"""
non_pi_elements = ["H", "He"]
pi_valencies = {
    "B": [1, 2],
    "N": [1, 2],
    "O": [1],
    "C": [1, 2, 3],
    "P": [1, 2, 3, 4],
    "S": [1, 3, 4, 5],
    "Si": [1, 2, 3],
}

# Standard definition of metallic elements: https://en.wikipedia.org/wiki/Metal
# (all semi-metals not included)
metals = [
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
]

# Covalent radii in picometers from https://en.wikipedia.org/wiki/Covalent_radius
_covalent_radii_pm = [
    31.0,
    28.0,
    128.0,
    96.0,
    84.0,
    76.0,
    71.0,
    66.0,
    57.0,
    58.0,
    166.0,
    141.0,
    121.0,
    111.0,
    107.0,
    105.0,
    102.0,
    106.0,
    102.0,
    203.0,
    176.0,
    170.0,
    160.0,
    153.0,
    139.0,
    161.0,
    152.0,
    150.0,
    124.0,
    132.0,
    122.0,
    122.0,
    120.0,
    119.0,
    120.0,
    116.0,
    220.0,
    195.0,
    190.0,
    175.0,
    164.0,
    154.0,
    147.0,
    146.0,
    142.0,
    139.0,
    145.0,
    144.0,
    142.0,
    139.0,
    139.0,
    138.0,
    139.0,
    140.0,
    244.0,
    215.0,
    207.0,
    204.0,
    203.0,
    201.0,
    199.0,
    198.0,
    198.0,
    196.0,
    194.0,
    192.0,
    192.0,
    189.0,
    190.0,
    187.0,
    175.0,
    187.0,
    170.0,
    162.0,
    151.0,
    144.0,
    141.0,
    136.0,
    136.0,
    132.0,
    145.0,
    146.0,
    148.0,
    140.0,
    150.0,
    150.0,
]

# Experimental bond lengths from https://cccbdb.nist.gov/diatomicexpbondx.asp
_bond_lengths = {"HH": 0.741, "FF": 1.412, "ClCl": 1.988, "II": 2.665}


_max_valances = {
    "H": 1,
    "He": 0,
    "B": 4,
    "C": 4,
    "N": 4,
    "O": 3,
    "F": 1,
    "Si": 4,
    "P": 6,
    "S": 6,
    "Cl": 4,
    "Br": 4,
    "I": 6,
    "Xe": 6,
    "Al": 4,
}
