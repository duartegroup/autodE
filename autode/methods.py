"""
Functions to get the high and low level electronic structure methods to use for example high-level methods would be
orca and Gaussian09 which can perform DFT/WF theory calculations, low level methods are for example xtb and mopac which
are non ab-initio methods and are therefore considerably faster
"""
from autode.config import Config
from autode.log import logger
from autode.wrappers.ORCA import orca
from autode.wrappers.MOPAC import mopac
from autode.wrappers.XTB import xtb
from autode.wrappers.G09 import g09
from autode.wrappers.NWChem import nwchem
from autode.exceptions import MethodUnavailable


def get_hmethod():
    """Get the high-level electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod):
    """
    if Config.hcode is not None:
        if Config.hcode.lower() == 'orca':
            method = orca
        elif Config.hcode.lower() == 'g09':
            method = g09
        elif Config.hcode.lower() == 'nwchem':
            method = nwchem
        else:
            logger.critical('Requested electronic structure code doesn\'t exist')
            raise MethodUnavailable

        method.set_availability()
        if not method.available:
            logger.critical('Requested electronic structure method is not available')
            raise MethodUnavailable

        return method
    else:
        # See if orca availaible, then Gaussian, then nwchem
        for method in [orca, g09, nwchem]:
            method.set_availability()
            if method.available:
                return method

        logger.critical('No electronic structure methods available')
        raise MethodUnavailable


def get_lmethod():
    """Get the low-level electronic structure theory method to use

    Returns:
        (autode.wrappers.base.ElectronicStructureMethod):
    """
    if Config.lcode is not None:
        if Config.lcode.lower() == 'xtb':
            method = xtb
        elif Config.lcode.lower() == 'mopac':
            method = mopac
        elif Config.lcode.lower() == 'orca':
            method = orca
            
        else:
            logger.critical('Requested electronic structure code doesn\'t exist')
            raise MethodUnavailable

        method.set_availability()
        if not method.available:
            logger.critical('Requested electronic structure method is not available')
            raise MethodUnavailable

        return method
    else:
        # See if xtb available, then mopac
        for method in [xtb, mopac]:
            method.set_availability()
            if method.available:
                return method

        logger.critical('No electronic structure methods available')
        raise MethodUnavailable
