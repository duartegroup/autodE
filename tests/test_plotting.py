from autode import plotting
from autode.units import KjMol
import os

here = os.path.dirname(os.path.abspath(__file__))

def test_plot_reaction_profile():
    #only tests the file is created with the right name
    os.chdir(os.path.join(here, 'data'))

    plotting.plot_reaction_profile(0.0, 10.0, -10.0, KjMol, 'name', True, True)

    assert os.path.exists('reaction_profile.png')

    os.remove('reaction_profile.png')
    os.chdir(here)
