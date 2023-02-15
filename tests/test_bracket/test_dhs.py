import numpy as np
from autode.bracket.dhs import DHSImagePair
from autode.bracket.dhs import AdaptiveBFGSMinimiser


def paraboloid_fn(arr):
    # z = (x-1.5)^2 / 5.0 + (y-2.0)^2 / 6.0
    x, y = tuple(arr)
    en = (x-1.5)**2 / 5.0 + (y-2.0)**2 / 6.0
    # dz/dx = 2(x-1.5)/5.0, dz/dy = 2(y-2.0)/6.0
    dz_dx = 2 * (x-1.5) / 5.0
    dz_dy = 2 * (y-2.0) / 6.0
    grad = np.array([dz_dx, dz_dy])
    return en, grad

def test_adaptive_bfgs():
    pass



