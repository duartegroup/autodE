"""
A more robust geometry optimiser, uses features from
multiple optimisation methods. (Similar to those
implemented in common QM packages)
"""

import numpy as np

from autode.opt.optimisers import RFOptimiser


class RobustOptimiser(RFOptimiser):
    def __init__(
        self,
        *args,
        init_trust: float = 0.1,
        min_trust: float = 0.01,
        max_trust: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, init_alpha=init_trust, **kwargs)
