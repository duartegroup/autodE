from autode.opt import get_ts_guess_constrained_opt
from autode.molecule import Molecule
from autode.config import Config
from autode.reactions import Substitution
from autode.wrappers.ORCA import ORCA
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_constrained_opt():
    os.chdir(os.path.join(here, 'data'))

    h3_xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0], ['H', 1.7, 0.0, 0.0]]
    mol = Molecule(name='h3', xyzs=h3_xyzs, mult=2)
    mol.method = ORCA()

    ts_guess = get_ts_guess_constrained_opt(mol=mol, distance_consts={(0, 1): 1.0}, reaction_class=Substitution,
                                            keywords=Config.ORCA.scan_keywords, name='template_ts_guess', product=mol)
    assert ts_guess.active_bonds == [(0, 1)]
    assert ts_guess.n_atoms == 3

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)
        if filename.endswith('.png'):
            os.remove(filename)

    os.chdir(here)
