from abc import ABC
from typing import Union, Collection
from autode.constants import Constants


class Unit(ABC):

    def __str__(self):
        return f'Unit({self.name})'

    def __repr__(self):
        return self.__str__()

    def lower(self):
        """Lower case name of the unit"""
        return self.name.lower()

    def __eq__(self, other):
        """Equality of two units"""
        return other.lower() in self.aliases

    def __init__(self,
                 name:       str,
                 conversion: float,
                 aliases:    Union[Collection, None] = None,
                 plot_name:  Union[str, None] = None):
        """
        Unit

        ----------------------------------------------------------------------
        Arguments:
            name (str):

            conversion (float): Conversion from default units to the new

        Keyword Arguments:
            aliases (list | set | tuple | None): Set of name aliases for this
                                                 unit

            plot_name (str | None): Name to use if this unit is used in a plot
        """

        self.name = name
        self.conversion = conversion

        self.aliases = [name.lower()]
        if aliases is not None:
            self.aliases += [alias.lower() for alias in aliases]

        self.plot_name = plot_name if plot_name is not None else name


class BaseUnit(Unit):
    """A unit in the base unit system, thus an identity conversion factor"""

    def __init__(self,
                 name: str,
                 aliases: Union[Collection, None] = None,
                 plot_name: Union[str, None] = None):

        super().__init__(name,
                         conversion=1.0,
                         aliases=aliases,
                         plot_name=plot_name)


ha = BaseUnit(name='Ha',
              aliases=['hartree', 'Eh'],
              plot_name='Ha')


ev = Unit(name='eV',
          conversion=Constants.ha_to_eV,
          aliases=['electron volt', 'electronvolt'],
          plot_name='eV')


# Upper case name to maintain backwards compatibility
kjmol = KjMol = Unit(name='kJ mol-1',
                     conversion=Constants.ha_to_kJmol,
                     aliases=['kjmol', 'kjmol-1', 'kj', 'kj mol'],
                     plot_name='kJ mol$^{-1}$')


kcalmol = KcalMol = Unit(name='kcal mol-1',
                         conversion=Constants.ha_to_kcalmol,
                         aliases=['kcalmol', 'kcalmol-1',
                                  'kcal', 'kcal mol'],
                         plot_name='kcal mol$^{-1}$')


rad = BaseUnit(name='rad',
               aliases=['radians'])


deg = Unit(name='°',
           conversion=Constants.rad_to_deg,
           aliases=['deg', 'degrees'])


ang = BaseUnit(name='Å',
               aliases=['ang', 'angstrom'])


a0 = Unit(name='bohr',
          conversion=Constants.ang_to_a0,
          aliases=['a0'])

nm = Unit(name='nm',
          conversion=Constants.ang_to_nm,
          aliases=['nanometer', 'nano meter'])

pm = Unit(name='pm',
          conversion=Constants.ang_to_pm,
          aliases=['picometer', 'pico meter'])

m = Unit(name='m',
         conversion=Constants.ang_to_m,
         aliases=['meter'])
