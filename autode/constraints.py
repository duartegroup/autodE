from typing import Optional, Dict, List
from autode.log import logger


class Constraints:

    def __init__(self,
                 distance:  Optional[Dict] = None,
                 cartesian: Optional[List] = None):
        """
        Arguments:
            distance (dict | None): Keys of: tuple(int) for two atom indexes
                                    and values of the distance in Å, or None

            cartesian (list(int) | None): List of atom indexes or None
        """
        self._distance = {}
        self._cartesian = []

        self.update(distance, cartesian)

    def __str__(self):
        """String of constraints"""
        string = ''

        if self.cartesian is not None:
            string += str(self.cartesian)

        if self.distance is not None:
            string += str({key: round(val, 3)
                           for key, val in self.distance.items()})

        return f'Constraints({string})'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _init_distance(dist_constraints: Dict):
        """Initialise the distance constraints"""
        assert type(dist_constraints) is dict
        distance = {}

        for key, val in dist_constraints.items():

            if len(set(key)) != 2:
                logger.warning('Can only set distance constraints between '
                               f'two distinct atoms. Had {key} - skipping')
                continue

            if float(val) < 0:
                raise ValueError('Negative distances are not valid'
                                 ' constraints!')

            distance[tuple(sorted(key))] = float(val)

        return distance

    @property
    def distance(self) -> Optional[dict]:
        return None if len(self._distance) == 0 else self._distance

    @distance.setter
    def distance(self, value: dict):
        """
        Set the distance constraints

        Arguments:
            value (dict): Dictionary keyed with atom indexes with values
                          as the distance between the two
        """
        self._distance = self._init_distance(value)

    @property
    def cartesian(self) -> list:
        return None if len(self._cartesian) == 0 else list(set(self._cartesian))

    @cartesian.setter
    def cartesian(self, value):
        """
        Set the Cartesian constraints using a list of atom indexes

        Arguments:
            value (list(int)): Atom indexes to fix in space
        """
        self._cartesian = [int(i) for i in value]

    @property
    def any(self):
        """Are there any constraints?"""
        return self.distance is not None or self.cartesian is not None

    def update(self,
               distance:  Optional[Dict] = None,
               cartesian: Optional[List] = None) -> None:
        """
        Update the current set of constraints with a new distance and or
        Cartesian set

        Arguments:
            distance (dict):

            cartesian (list):
        """

        if distance is not None:
            self._distance.update(self._init_distance(distance))

        if cartesian is not None:
            self._cartesian += cartesian

        return None
