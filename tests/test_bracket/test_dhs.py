import numpy as np
from autode.bracket.dhs import DHSImagePair
from autode.bracket.dhs import AdaptiveBFGSMinimiser


def paraboloid_fn(arr):
    # z = (x-1.5)^2 / 5.0 + (y+2.0)^2 / 6.0
    x, y = tuple(arr)
    en = (x-1.5)**2 / 5.0 + (y+2.0)**2 / 6.0
    # dz/dx = 2(x-1.5)/5.0, dz/dy = 2(y+2.0)/6.0
    dz_dx = 2 * (x-1.5) / 5.0
    dz_dy = 2 * (y+2.0) / 6.0
    grad = np.array([dz_dx, dz_dy])
    return en, grad


def test_adaptive_bfgs_paraboloid_minim():
    minim = AdaptiveBFGSMinimiser(
        fun=paraboloid_fn,
        x0=np.array([3.0, 4.0]),
        maxiter=200,
        gtol=1.0e-6,
    )
    res = minim.minimise()
    assert res['success']
    assert np.allclose(res['x'], [1.5, -2.0])



