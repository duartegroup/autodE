from autode.transition_states.ts_guess import get_ts_guess_constrained_opt
from autode.molecule import Molecule
from autode.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.atoms import Atom
from autode.wrappers.ORCA import orca
import os

here = os.path.dirname(os.path.abspath(__file__))
orca.available = True


def test_constrained_opt():
    os.chdir(os.path.join(here, 'data'))

    mol = Molecule(name='h3', mult=2, charge=0,
                   atoms=[Atom('H', 0.0, 0.0, 0.0),
                          Atom('H', 0.7, 0.0, 0.0),
                          Atom('H', 1.7, 0.0, 0.0)])

    Config.XTB.path = here       # A path that exists

    ts_guess = get_ts_guess_constrained_opt(reactant=ReactantComplex(mol),
                                            distance_consts={(0, 1): 1.0},
                                            method=orca,
                                            keywords=Config.ORCA.keywords.low_opt,
                                            name='template_ts_guess',
                                            product=ProductComplex(mol))
    assert ts_guess.n_atoms == 3

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)
        if filename.endswith('.png'):
            os.remove(filename)

    os.chdir(here)
