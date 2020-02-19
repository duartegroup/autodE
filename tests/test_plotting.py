from autode import plotting
from autode.molecule import Molecule
from autode.units import KjMol
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_plot_reaction_profile():
    # only tests the file is created with the right name
    os.chdir(os.path.join(here, 'data'))
    r = Molecule(name='reactant')
    p = Molecule(name='product')

    plotting.plot_reaction_profile(0.0, 10.0, -10.0, KjMol, reacs=[r], prods=[p], is_true_ts=True, ts_is_converged=True)

    assert os.path.exists('reaction_profile.png')

    os.remove('reaction_profile.png')
    os.chdir(here)
