from autode.constants import Constants


class Unit:
    pass


class KjMol(Unit):
    name = 'kJ mol$^{-1}$'
    conversion = Constants.ha2kJmol


class KcalMol(Unit):
    name = 'kcal mol$^{-1}$'
    conversion = Constants.ha2kcalmol
