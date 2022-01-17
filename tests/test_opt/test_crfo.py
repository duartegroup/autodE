import numpy as np
from autode.opt.optimisers import CRFOOptimiser
from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.utils import work_in_tmp_dir
from autode.wrappers.XTB import XTB
from autode.values import Distance
from ..testutils import requires_with_working_xtb_install


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_opt_with_constraints():

    water = Molecule(name='water', charge=0, mult=1,
                     atoms=[Atom('O', -0.00110, 0.36310,  0.00000),
                            Atom('H', -0.82500, -0.18190, 0.00000),
                            Atom('H',  0.82610, -0.18120, 0.00000)])

    water.constraints.distance = {(0, 1): Distance(1.1, units='Å')}

    assert np.isclose(water.distance(0, 1),
                      0.99,
                      atol=0.01)

    CRFOOptimiser.optimise(species=water,
                           method=XTB(),
                           init_alpha=0.05)  # Step size

    # Optimisation should generate an O-H distance *very* close to 1.1 Å
    assert np.isclose(water.distance(0, 1),
                      1.1000,
                      atol=1E-5)
