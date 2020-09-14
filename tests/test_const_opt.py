from autode.transition_states.ts_guess import get_ts_guess_constrained_opt
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

    mol = Molecule(name='h3', mult=2, charge=0,
                   atoms=[Atom('H', 0.0, 0.0, 0.0),
                          Atom('H', 0.7, 0.0, 0.0),
                          Atom('H', 1.7, 0.0, 0.0)])

    # Spoof an XTB install
    Config.XTB.path = here

    ts_guess = get_ts_guess_constrained_opt(reactant=ReactantComplex(mol),
                                            distance_consts={(0, 1): 1.0},
                                            method=XTB(),
                                            keywords=Config.XTB.keywords.low_opt,
                                            name='template_ts_guess',
                                            product=ProductComplex(mol))
    assert ts_guess.n_atoms == 3
