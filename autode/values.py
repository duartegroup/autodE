import numpy as np
from autode.log import logger
from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
from autode.units import (Unit,
                          ha, kjmol, kcalmol, ev,
                          ang, a0, nm, pm, m,
                          rad, deg)


class Value(ABC, float):
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
        """Copy this value, with it's units"""
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
            if other.units == self.units:
                return np.isclose(other, float(self))
            else:
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
        if self.units == units:
            return self

        try:
            units = next(imp_unit for imp_unit in self.implemented_units if
                         units.lower() in imp_unit.aliases)

        except StopIteration:
            raise TypeError(f'No viable unit conversion from {self.units} '
                            f'-> {units}')

        #                      Convert to the base unit, then to the new units
        return self.__class__(self * units.conversion / self.units.conversion,
                              units=units)

    def __init__(self, x,
                 units: Union[Unit, None] = None):
        """
        Value constructor

        Arguments:
            x (float | int):

        Keyword Arguments:
            units (autode.units.Unit | None):
        """
        float.__init__(float(x))
        self.units = units


class Energy(Value):
    """Type of energy in some units e.g. Potential, Free etc.
    defaults to Hartrees"""

    implemented_units = [ha, kcalmol, kjmol, ev]

    def __str__(self):
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

    def __str__(self):
        return f'FreeEnergy({round(self, 5)} {self.units.name})'


class Enthalpy(Energy):

    def __str__(self):
        return f'Enthalpy({round(self, 5)} {self.units.name})'


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

    def __str__(self):
        return f'Distance({round(self, 5)} {self.units.name})'

    def __init__(self, value, units=ang):
        super().__init__(value, units=units)


class Angle(Value):
    """Angle in some units, defaults to radians"""

    implemented_units = [rad, deg]

    def __str__(self):
        return f'Angle({round(self, 5)} {self.units.name})'

    def __init__(self, value, units=rad):
        super().__init__(value, units=units)


class Energies(list):

    def append(self, other: Energy):
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

    def _delta_to_electronic(self, energy_type):
        """
        Calculate X - E_elec, where X is perhaps a free energy or enthalpy

        Arguments:
            energy_type (autode.energies.Energy):

        Returns:
            (autode.energies.Energy | None):
        """
        try:
            # Select the final energy in this list with the correct type
            energy_with_type = next(e for e in reversed(self)
                                    if isinstance(e, energy_type))

            # and the corresponding electronic energy, calculated at the
            # same method, so the ∆ between is correct
            elec_energy = next(e for e in reversed(self)
                               if e.method_str == energy_with_type.method_str
                               and isinstance(e, PotentialEnergy))

            return Energy(energy_with_type - elec_energy)

        except StopIteration:
            logger.warning(f'Failed to calculate {energy_type.__name__} '
                           f'- {PotentialEnergy.__name__}')
            return None

    @property
    def h_cont(self) -> Union[Energy, None]:
        """
        Return the enthalpic contribution to the energy

        Returns:
             (autode.values.Energy | None): H_cont = H - E_elec
        """
        return self._delta_to_electronic(energy_type=Enthalpy)

    @property
    def g_cont(self) -> Union[Energy, None]:
        """
        Return the free energy contribution to the energy

        Returns:
             (autode.values.Energy | None): G_cont = G - E_elec
        """
        return self._delta_to_electronic(energy_type=FreeEnergy)

    def __init__(self, *args: Energy):
        """

        Arguments:
            *args (autode.values.Energy):
        """
        super().__init__(args)
