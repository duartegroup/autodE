from autode import plotting
from autode.molecule import Molecule
from autode.transition_states.transition_state import TransitionState
from autode.complex import ReactantComplex, ProductComplex
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.ts_guess import TSguess
from autode.units import KjMol
from autode.config import Config
import os

here = os.path.dirname(os.path.abspath(__file__))
Config.high_quality_plots = False


def test_plot_reaction_profile():
    # only tests the file is created with the right name
    os.chdir(os.path.join(here, 'data'))
    r = Molecule(name='reactant', smiles='C')
    p = Molecule(name='product', smiles='C')
    tsguess = TSguess(atoms=None, reactant=ReactantComplex(r), product=ProductComplex(p))
    ts = TransitionState(tsguess, bond_rearrangement=BondRearrangement())

    plotting.plot_reaction_profile(0.0, 10.0, -10.0, KjMol, reacs=[r], prods=[p], ts=ts)

    assert os.path.exists('reaction_profile.png')

    os.remove('reaction_profile.png')
    os.chdir(here)
