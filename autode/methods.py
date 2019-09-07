"""
Functions to get the high and low level electronic structure methods to use for example high-level methods would be
ORCA and PSI4 which can perform DFT/WF theory calculations, low level methods are for example XTB and MOPAC which
are non ab-initio methods
"""
from autode.config import Config
from autode.log import logger
from autode.wrappers.PSI4 import PSI4
from autode.wrappers.ORCA import ORCA
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.XTB import XTB


def get_hmethod():
    """
    Get the high-level electronic structure theory method to use
    :return:
    """
    method = PSI4
    if Config.hcode is not None:
        if Config.hcode.lower() == 'orca':
            method = ORCA
        elif Config.hcode.lower == 'psi4':
            method = PSI4
        else:
            logger.critical('Electronic structure code doesn\'t exist')
            exit()
    else:
        method = ORCA if ORCA.available else PSI4

    assert method.available is True
    return method


def get_lmethod():
    """
    Get the low-level electronic structure theory method to use
    :return:
    """
    method = XTB
    # if Config.lcode is not None:
    #     if Config.lcode.lower() == 'xtb':
    #         method = XTB
    #     elif Config.hcode.lower == 'mopac':
    #         method = MOPAC
    #     else:
    #         logger.critical('Electronic structure code doesn\'t exist')
    #         exit()
    # else:
    #     method = XTB if XTB.available else PSI4

    assert method.available is True
    return method
