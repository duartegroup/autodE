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
def test_xtb_opt_with_distance_constraint():

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
    assert np.isclose(water.distance(0, 1).to('Å'),
                      1.1000,
                      atol=1E-5)


def test_xtb_opt_with_cartesian_constraint():

    water = Molecule(name='water', charge=0, mult=1,
                     atoms=[Atom('O',  0.0507553,  0.439508, 0.0),
                            Atom('H', -0.8192601, -0.233597, 0.0),
                            Atom('H',  0.7685048, -0.205910, 0.0)])

    def distance_is_close_to_1p1():
        return np.isclose(water.distance(0, 1).to('Å'), 1.1,  atol=1E-3)

    assert distance_is_close_to_1p1()

    # Adding cartesian constraints to the 0 and 1st atoms and optimising
    # should not change the distance between them
    water.constraints.cartesian = [0, 1]

    CRFOOptimiser.optimise(species=water,
                           method=XTB(),
                           init_alpha=0.02)  # Step size

    assert distance_is_close_to_1p1()
