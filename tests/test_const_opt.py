from autode.transition_states.ts_guess import TSguess
from autode.species.molecule import Molecule
from autode.species.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.methods import XTB
from autode.atoms import Atom
from . import testutils
import os

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'constrained_opt.zip'))
def test_constrained_opt():

    # TODO: something reasonable here
    raise NotImplementedError
