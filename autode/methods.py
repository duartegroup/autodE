"""
Functions to get the high and low level electronic structure methods to use for example high-level methods would be
ORCA and Gaussian09 which can perform DFT/WF theory calculations, low level methods are for example XTB and MOPAC which
are non ab-initio methods and are therefore considerably faster
"""
from autode.config import Config
from autode.log import logger
from autode.wrappers.ORCA import ORCA
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.XTB import XTB
from autode.wrappers.G09 import G09
from autode.wrappers.NWChem import NWChem
from autode.exceptions import MethodUnavailable


def get_hmethod():
    """Get the high-level electronic structure theory method to use

    Returns:
        object: ElectronicStructureMethod
    """
    if Config.hcode is not None:
        if Config.hcode.lower() == 'orca':
            method = ORCA
        elif Config.hcode.lower() == 'g09':
            method = G09
        elif Config.hcode.lower() == 'nwchem':
            method = NWChem
        else:
            logger.critical('Electronic structure code doesn\'t exist')
            exit()
    else:
        method = ORCA

    method.set_availability()
    if not method.available:
        logger.critical('Electronic structure method is not available')
        raise MethodUnavailable

    return method


def get_lmethod():
    """Get the low-level electronic structure theory method to use

    Returns:
        object: ElectronicStructureMethod
    """
    if Config.lcode is not None:
        if Config.lcode.lower() == 'xtb':
            method = XTB
        elif Config.lcode.lower() == 'mopac':
            method = MOPAC
        else:
            logger.critical('Electronic structure code doesn\'t exist')
            exit()
    else:
        method = XTB

    method.set_availability()
    if not method.available:
        logger.critical('Electronic structure method is not available')
        raise MethodUnavailable

    return method
