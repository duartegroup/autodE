import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
from autode.units import (Unit,
                          ha, kjmol, kcalmol, ev,
                          ang, a0, nm, pm, m,
                          rad, deg)


class Value(ABC):
    """
    Abstract base class for a value with a defined set of units, along with
    perhaps other attributes and methods

    x = Value(0.0)
    """
    implemented_units = []

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        """String representation of this value"""

    def copy(self):
        return deepcopy(self)

    def _convert_to_float(self, other):
        """
        Convert another value to these units, do nothing if not a Value

        Arguments:
            other (autode.values.Value | float):

        Returns:
            (float):
        """

        if not isinstance(other, Value):
            return float(other)

        return other.to(self.units).x

    def __eq__(self, other):
        """Equality of two values, which may be in different units
        use default numpy close-ness to compare"""

        if isinstance(other, Value):
            if other.units == self.units:
                return np.isclose(other.x, self.x)
            else:
                return np.isclose(other.to(self.units).x, self.x)

        return np.isclose(other, self.x)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """Less than comparison operator"""

        if isinstance(other, Value):
            return self.x < other.to(self.units).x

        return self.x < other

    def __gt__(self, other):
        """Greater than comparison operator"""
        return not self.__lt__(other)

    def __le__(self, other):
        """Greater than or equal to comparison operator"""
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        """Less than or equal to comparison operator"""
        return self.__gt__(other) or self.__eq__(other)

    def __add__(self, other):
        """Add another value onto this one"""
        return self.__class__(self.x + self._convert_to_float(other),
                              units=self.units)

    def __mul__(self, other):
        """Multiply this value with another"""
        return self.__class__(self.x * self._convert_to_float(other),
                              units=self.units)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __float__(self):
        return self.x

    def __int__(self):
        return int(self.x)

    def __complex__(self):
        return complex(self.x)

    def to(self,
           units: Union[Unit, str]):
        """Convert this value to a new unit, returning a copy

        Arguments:
            units (autode.units.Unit | str):

        Returns:
            (autode.values.Value):

        Raises:
            (TypeError):
        """
        if isinstance(units, str):

            for imp_unit in self.implemented_units:
                if units.lower() in imp_unit.aliases:
                    units = imp_unit
                    break

        if units not in self.implemented_units:
            raise TypeError(f'No viable type conversion from {self.units} '
                            f'-> {units}')

        #                      Convert to the base unit, then to the new units
        return self.__class__(self.x * units.conversion/self.units.conversion,
                              units=units)

    def __init__(self, x,
                 units: Union[str, None] = None):
        """
        Value constructor

        Arguments:
            x (float | int):

        Keyword Arguments:
            units (autode.units.Unit | None):
        """
        self.x = float(x)
        self.units = units


class Energy(Value):
    """Type of energy in some units e.g. Potential, Free etc.
    defaults to Hartrees"""

    implemented_units = [ha, kcalmol, kjmol, ev]

    def __str__(self):
        return f'Energy({round(self.x, 5)} {self.units.name})'

    def __init__(self, value, units=ha):
        super().__init__(value, units=units)


class PlottedEnergy(Energy):

    def __init__(self, value, units=kcalmol, estimated=False):
        """
        An energy to be plotted on a reaction profile

        Arguments:
            value (float):

            estimated (bool): Has this energy been estimated rather than
                              calculated
        """
        super().__init__(value, units=units)

        self.estimated = estimated


class Distance(Value):
    """Distance in some units, defaults to Angstroms"""

    implemented_units = [ang, a0, pm, nm, m]

    def __str__(self):
        return f'Distance({round(self.x, 5)} {self.units.name})'

    def __init__(self, value, units=ang):
        super().__init__(value, units=units)


class Angle(Value):
    """Distance in some units, defaults to Angstroms"""

    implemented_units = [rad, deg]

    def __str__(self):
        return f'Angle({round(self.x, 5)} {self.units.name})'

    def __init__(self, value, units=rad):
        super().__init__(value, units=units)
