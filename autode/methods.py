from typing import Optional
from autode.log import logger
from autode.wrappers.QChem import QChem
from autode.config import Config
from autode.exceptions import MethodUnavailable

"""
Functions to get the high and low level electronic structure methods to use 
for example high-level methods would be orca and Gaussian09 which can perform 
DFT/WF theory calculations, low level methods are, for example, xtb and mopac 
which are fast non ab-initio methods
"""

high_level_method_names = ['orca', 'g09', 'g16', 'nwchem', 'qchem']
low_level_method_names = ['xtb', 'mopac']


def method_or_default_lmethod(method: Optional["autode.wrappers.methods.Method"]):
    """
    Return a method if one is defined but default to a low-level method if
    if it is None.

    ---------------------------------------------------------------------------
    Arguments:
        method: Method or None

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method
    """
    if method is None:
        method = get_lmethod()
        logger.info(f'Using the default low-level method {method}')

    return method


def method_or_default_hmethod(method: Optional["autode.wrappers.methods.Method"]):
    """
    Return a method if one is defined but default to a high-level method if
    if it is None.

    ---------------------------------------------------------------------------
    Arguments:
        method: Method or None

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method
    """
    if method is None:
        method = get_hmethod()
        logger.info(f'Using the default high-level method {method}')

    return method


def get_hmethod() -> "autode.wrappers.methods.Method":
    """Get the 'high-level' electronic structure theory method to use

    ---------------------------------------------------------------------------
    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): High-level method
    """

    h_methods = _methods_list(high_level=True)

    if Config.hcode is not None:
        return get_defined_method(name=Config.hcode, possibilities=h_methods)
    else:
        return get_first_available_method(h_methods)


def get_lmethod() -> "autode.wrappers.methods.Method":
    """Get the 'low-level' electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Low-level method
    """

    all_methods = _methods_list(high_level=False)

    if Config.lcode is not None:
        return get_defined_method(name=Config.lcode, possibilities=all_methods)
    else:
        return get_first_available_method(all_methods)


def get_first_available_method(possibilities) -> "autode.wrappers.methods.Method":
    """
    Get the first electronic structure method that is available in a list of
    possibilities.

    ---------------------------------------------------------------------------
    Arguments:
        possibilities (list(autode.wrappers.base.ElectronicStructureMethod)):

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod): Method

    Raises:
        (autode.exceptions.MethodUnavailable):
    """
    for method in possibilities:

        if method.is_available:
            return method

    raise MethodUnavailable('No electronic structure methods available')


def get_defined_method(name, possibilities) -> "autode.wrappers.methods.Method":
    """
    Get an electronic structure method defined by it's name.

    ---------------------------------------------------------------------------
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

            if method.is_available:
                return method

            else:
                err_str = (f'Electronic structure method *{name}* is not '
                           f'available. Check that {method.name} exists in a '
                           f'directory present in $PATH, or set '
                           f'ade.Config.{method.__name__}.path')

                raise MethodUnavailable(err_str)

    raise MethodUnavailable('Requested code does not exist')


def _methods_list(high_level=False):
    from autode.wrappers.G09 import G09
    from autode.wrappers.G16 import G16
    from autode.wrappers.NWChem import NWChem
    from autode.wrappers.ORCA import ORCA
    from autode.wrappers.QChem import QChem
    from autode.wrappers.MOPAC import MOPAC
    from autode.wrappers.XTB import XTB

    all_methods = [XTB(), MOPAC(), ORCA(), G16(), G09(), NWChem(), QChem()]

    return all_methods[2:] if high_level else all_methods
