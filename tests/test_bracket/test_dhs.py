import os
import numpy as np
import pytest

from autode import Molecule
from autode.methods import XTB
from autode.utils import work_in
from autode.geom import calc_rmsd
from autode.bracket.dhs import DHS
from ..testutils import requires_with_working_xtb_install

here = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(here, "data")
# todo replace with zip later


@requires_with_working_xtb_install
@work_in(datadir)
def test_dhs_diels_alder():
    set_dist_tol = 1.0  # angstrom

    reactant = Molecule("da_reactant.xyz")
    product = Molecule("da_product.xyz")
    # optimized with ORCA using xTB method
    true_ts = Molecule("da_ts_orca_xtb.xyz")

    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=2000,
        reduction_factor=0.05,  # fails with high reduction fact?
        dist_tol=set_dist_tol,
        optimiser="BFGS",
    )

    dhs.calculate(method=XTB(), n_cores=1)
    assert dhs.converged
    peak = dhs.get_peak_species()

    rmsd = calc_rmsd(peak.coordinates, true_ts.coordinates)
    # Euclidean distance = rmsd * sqrt(n_atoms)
    distance = rmsd * np.sqrt(peak.n_atoms)

    # the true TS must be within the last two DHS images,
    # therefore the distance must be less than the distance
    # tolerance (assuming fairly linear PES near TS)

    assert distance < set_dist_tol
