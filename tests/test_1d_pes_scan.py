from autode.pes_1d import get_ts_guess_1dpes_scan
from autode.molecule import Molecule
from autode.config import Config
from autode.reactions import Substitution
from autode.wrappers.ORCA import ORCA
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_get_ts_guess_1dscan():
    os.chdir(os.path.join(here, 'data'))

    h3_xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0], ['H', 1.7, 0.0, 0.0]]
    mol = Molecule(name='h3', xyzs=h3_xyzs, mult=2)
    mol.method = ORCA

    # May not actually be available, but the .out files are included so this should work
    mol.method.available = True

    ts_guess = get_ts_guess_1dpes_scan(mol=mol, active_bond=(1, 2), n_steps=5, name='h3_subst',
                                       reaction_class=Substitution, method=ORCA, keywords=Config.ORCA.scan_keywords,
                                       products = [mol], delta_dist=-0.3, active_bonds_not_scanned=[(0, 1)])

    assert len(ts_guess.active_bonds) == 2
    assert os.path.exists('h3_1dscan.png')

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)
        if filename.endswith('.png'):
            os.remove(filename)
    
    os.chdir(here)
