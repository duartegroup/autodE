from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.G09 import G09
from autode.wrappers.G16 import G16
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.NWChem import NWChem
from autode.wrappers.ORCA import ORCA
from autode.wrappers.XTB import XTB
from autode.config import Config
from autode.exceptions import MethodUnavailable

"""
Functions to get the high and low level electronic structure methods to use 
for example high-level methods would be orca and Gaussian09 which can perform 
DFT/WF theory calculations, low level methods are for example xtb and mopac 
which are non ab-initio methods and are therefore considerably faster
"""

high_level_method_names = ['orca', 'g09', 'g16', 'nwchem']
low_level_method_names = ['xtb', 'mopac']


def get_hmethod() -> ElectronicStructureMethod:
    """Get the 'high-level' electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): High-level method
    """
    h_methods = [ORCA(), G09(), NWChem(), G16()]

    if Config.hcode is not None:
        return get_defined_method(name=Config.hcode, possibilities=h_methods)
    else:
        return get_first_available_method(h_methods)


def get_lmethod() -> ElectronicStructureMethod:
    """Get the 'low-level' electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Low-level method
    """
    all_methods = [XTB(), MOPAC(), ORCA(), G16(), G09(), NWChem()]

    if Config.lcode is not None:
        return get_defined_method(name=Config.lcode, possibilities=all_methods)
    else:
        return get_first_available_method(all_methods)


def get_first_available_method(possibilities) -> ElectronicStructureMethod:
    """
    Get the first electronic structure method that is available in a list of
    possibilities.

    Arguments:
        possibilities (list(autode.wrappers.base.ElectronicStructureMethod)):

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method

    Raises:
        (autode.exceptions.MethodUnavailable):
    """
    for method in possibilities:

        if method.available:
            return method

    raise MethodUnavailable('No electronic structure methods available')


def get_defined_method(name, possibilities) -> ElectronicStructureMethod:
    """
    Get an electronic structure method defined by it's name.

    Arguments:
        name (str):
        possibilities (list(autode.wrappers.base.ElectronicStructureMethod)):

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method

    Raises:
        (autode.exceptions.MethodUnavailable):
    """

    for method in possibilities:
        if method.name.lower() == name.lower():

            if method.available:
                return method

            else:
                err_str = (f'Electronic structure method *{name}* is not '
                           f'available. Check that {method.name} exists in a '
                           f'directory present in $PATH, or set '
                           f'ade.Config.{method.__name__}.path')

                raise MethodUnavailable(err_str)

    raise MethodUnavailable('Requested code does not exist')
