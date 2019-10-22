from autode.pes_2d import get_ts_guess_2d
from autode.molecule import Molecule
from autode.config import Config
from autode.reactions import Dissociation
from autode.wrappers.ORCA import ORCA
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_get_ts_guess_2dscan():
    os.chdir(os.path.join(here, 'data'))

    h4_xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0], ['H', 0.0, -0.7, 0.0], ['H', 0.7, -0.7, 0.0]]
    mol = Molecule(name='h4', xyzs=h4_xyzs, mult=1)
    mol.method = ORCA
    product_xyzs = [['H', 0.0, 0.1, 0.0], ['H', 0.7, 0.1, 0.0], ['H', 0.0, -0.8, 0.0], ['H', 0.7, -0.8, 0.0]]
    product = Molecule(name='h4product', xyzs=product_xyzs, mult=1)

    # May not actually be available, but the .out files are included so this should work
    mol.method.available = True

    ts_guess = get_ts_guess_2d(mol=mol, product=mol, active_bond1=(0, 2), active_bond2=(1, 3), n_steps=2, name='h4',
                               reaction_class=Dissociation, method=ORCA, keywords=Config.ORCA.scan_keywords,
                               products = [product], delta_dist1=0.2, delta_dist2=0.2)

    assert len(ts_guess.active_bonds) == 2
    assert os.path.exists('h4_2dscan_ORCA.png')

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(('.inp', '.png', '.xyz')):
            os.remove(filename)

    os.chdir(here)
