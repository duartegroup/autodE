from abc import ABC
from typing import Union, Collection
from autode.constants import Constants


class Unit(ABC):

    def __str__(self):
        return f'Unit({self.name})'

    def __repr__(self):
        return self.__str__()

    def lower(self):
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


ha = Unit(name='Ha',
          conversion=1.0,
          aliases=['hartree', 'Eh'],
          plot_name='Ha')


ev = Unit(name='eV',
          conversion=27.2114,
          aliases=['electron volt', 'electronvolt'],
          plot_name='eV')


# Upper case name to maintain backwards compatibility
kjmol = KjMol = Unit(name='kj mol-1',
                     conversion=Constants.ha_to_kJmol,
                     aliases=['kjmol', 'kjmol-1', 'kj', 'kj mol'],
                     plot_name='kJ mol$^{-1}$')


kcalmol = KcalMol = Unit(name='kcal mol-1',
                         conversion=Constants.ha_to_kcalmol,
                         aliases=['kcalmol', 'kcalmol-1',
                                  'kcal', 'kcal mol'],
                         plot_name='kcal mol$^{-1}$')


deg = Unit(name='degrees',
           conversion=57.2958,    # rad -> deg
           aliases=['deg', '°'])


rad = Unit(name='radians',
           conversion=1.0,
           aliases=['rad'])


ang = Unit(name='angstrom',
           conversion=1.0,
           aliases=['ang'])


a0 = Unit(name='bohr',
          conversion=Constants.a0_to_ang,
          aliases=['a0'])

nm = Unit(name='nm',
          conversion=10.0,
          aliases=['nanometer', 'nano meter'])

pm = Unit(name='pm',
          conversion=1E-2,
          aliases=['picometer', 'pico meter'])

m = Unit(name='m',
         conversion=1E-10,
         aliases=['meter'])
