"""
Constrained optimisation with trust radius model
"""
import numpy as np
from typing import Union

from autode.log import logger
from autode.values import GradientRMS, Distance
from autode.opt.optimisers.crfo import CRFOptimiser


class TRMOptimiser(CRFOptimiser):
    """"""
