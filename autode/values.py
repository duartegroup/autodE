import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Type, Optional
from copy import deepcopy
from collections.abc import Iterable
from autode.log import logger
from autode.units import (Unit,
                          ha, kjmol, kcalmol, ev,
                          ang, a0, nm, pm, m,
                          rad, deg,
                          wavenumber, hz,
                          amu, kg, m_e,
                          amu_ang_sq, kg_m_sq,
                          ha_per_ang, ha_per_a0, ev_per_ang)


def _to(value,
        units: Union[Unit, str]):
    """Convert a value or value array to a new unit and return a copy

    Arguments:
        value (autode.values.Value | autode.values.ValueArray):
        units (autode.units.Unit | str):

    Returns:
        (autode.values.Value):
    """
    if value.units == units:
        return value

    try:
        units = next(imp_unit for imp_unit in value.implemented_units if
                     units.lower() in imp_unit.aliases)

    except StopIteration:
        raise TypeError(f'No viable unit conversion from {value.units} '
                        f'-> {units}')

    #                      Convert to the base unit, then to the new units
    return value.__class__(value * units.conversion / value.units.conversion,
                           units=units)


def _units_init(value,
                units: Union[Unit, str, None]):
    """Initialise the units of this value

    Arguments:
        units (Unit | str | None)

    Raises:
        (ValueError): If this is not a valid unit for this value
    """
    if units is None:
        return None

    try:
        return next(unit for unit in value.implemented_units if
                    units.lower() in unit.aliases)

    except StopIteration:
        raise ValueError(f'{units} is not a valid unit for '
                         f'{repr(value)}. Only {value.implemented_units} '
                         f'are implemented')


class Value(ABC, float):
    """
    Abstract base class for a value with a defined set of units, along with
    perhaps other attributes and methods

    x = Value(0.0)
    """
    implemented_units = []

    @abstractmethod
    def __repr__(self):
        """Internal representation of this value"""

    def __str__(self):
        """String representation of this value"""
        # NOTE: as this may be used in f-strings just return the value as a str
        return str(float(self))

    def copy(self):
        """Copy this value, with its units"""
        return deepcopy(self)

    def _other_same_units(self, other):
        """
        Convert another value to these units, do nothing if not a Value

        Arguments:
            other (autode.values.Value | float):

        Returns:
            (float):
        """
        if not isinstance(other, Value):
            return other

        return other.to(self.units)

    def __eq__(self, other):
        """Equality of two values, which may be in different units
        use default numpy close-ness to compare"""

        if other is None:
            return False

        if isinstance(other, Value):
            return np.isclose(other.to(self.units), float(self))

        return np.isclose(other, float(self))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """Less than comparison operator"""

        if isinstance(other, Value):
            return float(self) < other.to(self.units)

        return float(self) < other

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
        if isinstance(other, np.ndarray):
            return other + float(self)

        return self.__class__(float(self) + self._other_same_units(other),
                              units=self.units)

    def __mul__(self, other):
        """Multiply this value with another"""
        if isinstance(other, np.ndarray):
            return other * float(self)

        return self.__class__(float(self) * self._other_same_units(other),
                              units=self.units)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def to(self, units):
        """Convert this value to a new unit, returning a copy

        Arguments:
            units (autode.units.Unit | str):

        Returns:
            (autode.values.Value):

        Raises:
            (TypeError):
        """
        return _to(self, units)

    def __init__(self, x,
                 units: Union[Unit, str, None] = None):
        """
        Value constructor

        Arguments:
            x (float | int):

        Keyword Arguments:
            units (autode.units.Unit | None):
        """

        float.__init__(float(x))
        self.units = _units_init(self, units)


class Energy(Value):
    """Type of energy in some units e.g. Potential, Free etc.
    defaults to Hartrees"""

    implemented_units = [ha, kcalmol, kjmol, ev]

    def __repr__(self):
        return f'Energy({round(self, 5)} {self.units.name})'

    def __eq__(self, other):
        """Is an energy equal to another? Compares only the value, with
        implicit unit conversion"""

        # A PotentialEnergy is not equal to a FreeEnergy, for example
        if isinstance(other, Energy) and not isinstance(other, self.__class__):
            return False

        return super().__eq__(other)

    def __init__(self,
                 value,
                 units: Unit = ha,
                 method=None,
                 keywords=None):
        """
        Energy as a value. Has a method_str attribute which is set using a
        method used to calculate the energy along with any keywords e.g.
        PBE0/def2-SVP used to calculate it

        ----------------------------------------------------------------------
        Arguments:

            value: Float-able number

            units (autode.units.Unit): Unit type, allowing conversion

            method (autode.wrappers.base.Method):

            keywords (autode.wrappers.keywords.Keywords | None): Set of
                     keywords which this energy has been calculated at
        """
        super().__init__(value, units=units)

        self.method_str = f'{method.name} ' if method is not None else 'unknown'
        self.method_str += keywords.bstring if keywords is not None else ''


class PotentialEnergy(Energy):
    """Potential electronic energy (0 K, no zero-point energy)"""


class FreeEnergy(Energy):

    def __repr__(self):
        return f'FreeEnergy({round(self, 5)} {self.units.name})'


class Enthalpy(Energy):

    def __repr__(self):
        return f'Enthalpy({round(self, 5)} {self.units.name})'


class EnthalpyCont(Energy):

    def __repr__(self):
        return f'G_cont({round(self, 5)} {self.units.name})'


class FreeEnergyCont(Energy):

    def __repr__(self):
        return f'H_cont({round(self, 5)} {self.units.name})'


class Energies(list):
    """List of energies on an identical geometry/structure"""

    def append(self, other: Energy) -> None:
        """
        Add another energy to this list, if it does not already appear

        Arguments:
             other (autode.values.Energy):
        """

        for item in self:
            if other == item:
                logger.warning(f'Not appending {other} to the energies - '
                               f'already present')
                return

        return super().append(other)

    def last(self, energy_type: Type[Energy]) -> Optional[Energy]:
        """
        Return the last instance of a particular energy type in these list
        of energies

        Arguments:
            energy_type (Energy):

        Returns:
            (autode.values.Energy | None):
        """

        try:
            return next(energy for energy in reversed(self)
                        if isinstance(energy, energy_type))

        except StopIteration:
            return None

    def __init__(self, *args: Energy):
        """

        Arguments:
            *args (autode.values.Energy):
        """
        super().__init__(args)


class PlottedEnergy(Energy):

    def __eq__(self, other):
        """Is an energy equal to another? Compares only the value, not
        whether they are estimated"""
        return super().__eq__(other)

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

    def __repr__(self):
        return f'Distance({round(self, 5)} {self.units.name})'

    def __init__(self, value, units=ang):
        super().__init__(value, units=units)


class Angle(Value):
    """Angle in some units, defaults to radians"""

    implemented_units = [rad, deg]

    def __repr__(self):
        return f'Angle({round(self, 5)} {self.units.name})'

    def __init__(self, value, units=rad):
        super().__init__(value, units=units)


class Frequency(Value):

    implemented_units = [wavenumber, hz]

    @property
    def is_imaginary(self) -> bool:
        """Imaginary frequencies are quoted as negative for simplicity"""
        return self < 0

    @property
    def real(self) -> 'Frequency':
        """
        A frequencies real (positive) value

        Returns:
            (autode.values.Frequency):
        """
        return self * -1 if self.is_imaginary else self

    def __repr__(self):
        return f'Frequency({round(self, 5)} {self.units.name})'

    def __init__(self, value, units=wavenumber):
        super().__init__(value, units=units)


class Mass(Value):

    implemented_units = [amu, kg, m_e]

    def __repr__(self):
        return f'Mass({round(self, 5)} {self.units.name})'

    def __init__(self, value, units=amu):
        super().__init__(value, units=units)


class ValueArray(ABC, np.ndarray):
    """
    Abstract base class for an array of values, e.g. gradients or a Hessian
    """
    implemented_units = []

    @abstractmethod
    def __repr__(self):
        """String representation of this value array"""

    def __eq__(self, other):
        """Define equality for a valuearray, with implicit type conversion"""

        if (other is None
                or not hasattr(other, 'shape')
                or other.shape != self.shape):
            return False

        if isinstance(other, ValueArray):
            other = other.to(self.units)

        return np.allclose(self, other)

    def __new__(cls,
                input_array: np.ndarray,
                units: Union[Unit, str, None] = None):
        """
        Initialise a ValueArray from a numpy array, or another ValueArray

        Arguments:
            input_array (np.ndarray | autode.values.ValueArray):
            units (autode.units.Unit | str):

        Returns:
            (autode.values.ValueArray):
        """

        arr = np.asarray(input_array).view(cls)

        if isinstance(input_array, ValueArray) and units is None:
            arr.units = input_array.units
        else:
            arr.units = _units_init(cls, units)

        return arr

    def to(self, units):
        """
        Convert this array to a new unit, returning a copy

        Arguments:
            units (autode.units.Unit | str):

        Returns:
            (autode.values.ValueArray):

        Raises:
            (TypeError):
        """
        return _to(self, units)

    def __array_finalize__(self, obj):
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        if obj is None:
            return
        self.units = getattr(obj, 'units', None)


class Coordinate(ValueArray):

    implemented_units = [ang, a0, nm, pm, m]

    def __repr__(self):
        return f'Coordinate({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls, *args, units=ang):

        if len(args) == 3:
            return super().__new__(cls, np.asarray(args), units)

        elif (len(args) == 1
              and isinstance(args[0], Iterable)
              and len(args[0]) == 3):
            # e.g. a numpy array or list of three elements
            return super().__new__(cls, np.asarray(args[0]), units)

        else:
            raise ValueError('Coordinate must be a 3 component vector, got '
                             f'{len(args)} components')

    @property
    def x(self):
        """x component in Cartesian space"""
        return self[0]

    @property
    def y(self):
        """y component in Cartesian space"""
        return self[1]

    @property
    def z(self):
        """z component in Cartesian space"""
        return self[2]


class Coordinates(ValueArray):

    implemented_units = [ang, a0, nm, pm, m]

    def __repr__(self):
        return f'Coordinates({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls, input_array, units=ang):
        return super().__new__(cls, input_array.reshape(-1, 3), units)


class Gradients(ValueArray):

    implemented_units = [ha_per_ang, ha_per_a0, ev_per_ang]

    def __repr__(self):
        return f'Gradients({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls,  input_array, units=ha_per_ang):
        return super().__new__(cls, input_array, units)


class MomentOfInertia(ValueArray):

    implemented_units = [amu_ang_sq, kg_m_sq]

    def __repr__(self):
        return f'I({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls,  input_array, units=amu_ang_sq):
        return super().__new__(cls, input_array, units)
