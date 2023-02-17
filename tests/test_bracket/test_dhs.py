import os
import numpy as np
import pytest

from autode import Molecule
from autode.methods import XTB
from autode.utils import work_in
from autode.geom import calc_rmsd
from autode.bracket.dhs import (AdaptiveBFGSMinimiser,
                                _minimise, DHS)
from ..testutils import requires_with_working_xtb_install

here = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(here, "data")
# todo replace with zip later

# start of tests for optimiser =>


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
        options={'maxiter':200, 'gtol':1.0e-6}
    )
    res = minim.minimise()
    assert res['success']
    assert np.allclose(res['x'], [1.5, -2.0])


@pytest.mark.parametrize('method', ['adaptBFGS', 'BFGS', 'CG'])
def test_common_minimise_interface(method):
    res = _minimise(
        fun=paraboloid_fn,
        x0=np.array([3.0, 4.0]),
        method=method,
        options={'maxiter': 200, 'gtol': 1.0e-6}
    )
    assert res['success']
    assert np.allclose(res['x'], [1.5, -2.0])

# <= end of tests for optimisers


@requires_with_working_xtb_install
@work_in(datadir)
def test_dhs_diels_alder():
    set_dist_tol = 0.6  # angstrom

    reactant = Molecule('da_reactant.xyz')
    product = Molecule('da_product.xyz')
    # optimized with ORCA using xTB method
    true_ts = Molecule('da_ts_orca_xtb.xyz')

    dhs = DHS(initial_species=reactant,
              final_species=product,
              maxiter=2000,
              reduction_factor=0.05,  # fails with high reduction fact?
              dist_tol=set_dist_tol,
              optimiser='BFGS')

    dhs.calculate(method=XTB(), n_cores=1)
    peak = dhs.get_peak_species()
    peak.print_xyz_file('peak.xyz')

    rmsd = calc_rmsd(peak.coordinates, true_ts.coordinates)
    # Euclidean distance = rmsd * sqrt(n_atoms)
    distance = rmsd * np.sqrt(peak.n_atoms)

    # the true TS must be within the last two DHS images,
    # therefore the distance must be less than the distance
    # tolerance (assuming fairly linear PES near TS)

    assert distance < set_dist_tol

