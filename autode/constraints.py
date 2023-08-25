from collections.abc import MutableMapping
from typing import Optional, Dict, List
from copy import deepcopy

from autode.values import Distance, Angle, deg
from autode.log import logger

class Constraints:
    def __init__(
        self, distance: Optional[Dict] = None, cartesian: Optional[List] = None, angular: Optional[Dict] = None
    ):
        """
        Constrained distances, positions and dihedrals

        -----------------------------------------------------------------------
        Arguments:
            distance (dict | None): Keys of: tuple(int) for two atom indexes
                                    and values of the distance in Å, or None
            cartesian (list(int) | None): List of atom indexes or None
            angular (dict | None): Keys of: tuple(int) for four atom indexes
                                    and values of the angels, or None
        """

        # addition of _angular attribute
        self._distance = DistanceConstraints()
        self._cartesian: List[int] = []
        self._angular = AngleConstraints()

        self.update(distance, cartesian, angular)

    def __str__(self):
        """String of constraints"""
        string = ""

        # Update for angular
        if self.cartesian is not None:
            string += str(self.cartesian)
        if self.distance is not None:
            string += str({key: round(val, 3) for key, val in self.distance.items()})
        if self.angular is not None:
            string += str({key: round(val, 3) for key, val in self.angular.items()})
        
        return f"Constraints({string})"

    def __repr__(self):
        return self.__str__()

    # Required methods/properties

    @property
    def distance(self) -> Optional["DistanceConstraints"]:
        return None if len(self._distance) == 0 else self._distance

    @distance.setter
    def distance(self, value: Optional[dict]):
        """
        Set the distance constraints

        -----------------------------------------------------------------------
        Arguments:
            value (dict | None): Dictionary keyed with atom indexes with values
                                  as the distance between the two
        """
        if value is None:
            self._distance.clear()
        else:
            self._distance = DistanceConstraints(value)

    @property
    def n_distance(self) -> int:
        """Number of distance constraints"""
        return len(self._distance)

    @property
    def cartesian(self) -> Optional[list]:
        """Cartesian constraints"""
        return None if len(self._cartesian) == 0 else list(set(self._cartesian))

    @cartesian.setter
    def cartesian(self, value: Optional[List[int]]):
        """
        Set the Cartesian constraints using a list of atom indexes

        -----------------------------------------------------------------------
        Arguments:
            value (list(int) | None): Atom indexes to fix in space
        """
        if value is None:
            self._cartesian.clear()
        else:
            self._cartesian = [int(i) for i in value]

    @property
    def n_cartesian(self) -> int:
        """Number of cartesian constraints"""
        return len(self._cartesian)

    # Add methods/properties for angular

    @property
    def angular(self) -> Optional["AngleConstraints"]:
        return None if len(self._angular) == 0 else self._angular

    @angular.setter
    def angular(self, value: Optional[dict]):
        """
        Set the angular constraints

        -----------------------------------------------------------------------
        Arguments:
            value (dict | None): Dictionary keyed with atom indexes with values
                                 as the angular between the three or four indices
        """
        if value is None:
            self._angular.clear()
        else:
            self._angular = AngleConstraints(value)

    @property
    def n_angular(self) -> int:
        """Number of angular constraints"""
        return len(self._angular)

    @property
    def any(self) -> bool:
        """Are there any constraints?"""
        return self.distance is not None or self.cartesian is not None or self.angular is not None

    def update(
        self,
        distance: Optional[dict] = None,
        cartesian: Optional[List[int]] = None,
        angular: Optional[dict] = None,
    ) -> None:
        """
        Update the current set of constraints with a new distance, Cartesian and or
        angular set

        -----------------------------------------------------------------------
        Arguments:
            distance (dict | None):
            cartesian (list | None):
            angular (dict | None):
        """
        if distance is not None:
            self._distance.update(DistanceConstraints(distance))

        if cartesian is not None:
            self._cartesian += cartesian

        if angular is not None:
            self._angular.update(AngleConstraints(angular))
        return None

    def copy(self) -> "Constraints":
        return deepcopy(self)

class DistanceConstraints(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self._store[self._key_transform(key)]

    def __delitem__(self, key):
        del self._store[self._key_transform(key)]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    @staticmethod
    def _key_transform(key):
        """Transform the key to a sorted tuple"""
        return tuple(sorted(key))

    def __setitem__(self, key, value):
        """
        Set a key-value pair in the dictionary

        -----------------------------------------------------------------------
        Arguments:
            key (tuple(int)): Pair of atom indexes

            value (int | float): Distance
        """
        try:
            n_unique_atoms = len(set(key))

        except TypeError:
            raise ValueError(f"Cannot set a key with {key}, must be iterable")

        if n_unique_atoms != 2:
            logger.error(
                "Tried to set a distance constraint with a key: "
                f"{key}. Must be a unique pair of atom indexes"
            )
            return

        if float(value) <= 0:
            raise ValueError("Negative distances are not valid constraints!")

        if any(int(atom_idx) < 0 for atom_idx in key):
            raise ValueError(
                "Distance constraint key must be an atom index "
                f"pair but had: {key} which cannot be valid (<0)"
            )

        self._store[self._key_transform(key)] = Distance(value)

    def copy(self) -> "DistanceConstraints":
        return deepcopy(self)

class AngleConstraints(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._store = {}
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self._store[self._key_transform(key)]

    def __delitem__(self, key):
        del self._store[self._key_transform(key)]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    @staticmethod
    def _key_transform(key):
        """Transform the key to a sorted tuple"""
        return tuple(sorted(key))

    def __setitem__(self, key, value):
        """
        Set a key-value pair in the dictionary

        ------------------------------------------------------------------------
        Arguments:
            key (tuple(int)): Tuple of 3 or 4 atom indexes

            value (int | float): angular or Dihedral angular
        """
        try:
            n_unique_atoms = len(set(key))
        except TypeError as e:
            raise ValueError(f"Cannot set a key with {key}, must be iterable") from e

        if n_unique_atoms not in [3, 4]:
            logger.error(
                "Tried to set an angular/dihedral constraint with a key: "
                f"{key}. Must be a unique tuple of 3 or 4 atom indexes"
            )
            return

        if any(int(atom_idx) < 0 for atom_idx in key):
            raise ValueError(
                "Angular constraint key must be an atom index "
                f"quad but had: {key} which cannot be valid (<0)"
            )

        self._store[self._key_transform(key)] = Angle(value, units=deg)

    def copy(self) -> "AngleConstraints":
        return deepcopy(self)
