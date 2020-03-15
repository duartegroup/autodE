from autode.pes_1d import get_ts_guess_1d
from autode.molecule import Molecule
from autode.config import Config
from autode.reactions import Substitution
from autode.wrappers.ORCA import ORCA
import os

here = os.path.dirname(os.path.abspath(__file__))

Config.ORCA.path = os.path.join(here, 'test_1d_pes_scan.py')
Config.XTB.path = os.path.join(here, 'test_1d_pes_scan.py')
Config.MOPAC.path = os.path.join(here, 'test_1d_pes_scan.py')
Config.MOPAC.path_to_licence = os.path.join(here, 'test_1d_pes_scan.py')
Config.G09.path = os.path.join(here, 'test_1d_pes_scan.py')


def test_get_ts_guess_1dscan():
    os.chdir(os.path.join(here, 'data'))

    h3_xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0], ['H', 1.7, 0.0, 0.0]]
    mol = Molecule(name='h3', xyzs=h3_xyzs, mult=2)
    method = ORCA()

    ts_guess = get_ts_guess_1d(mol=mol, product=mol, active_bond=(1, 2), n_steps=5, name='h3_subst',
                               reaction_class=Substitution, method=method, keywords=Config.ORCA.scan_keywords,
                               final_dist=-0.3, active_bonds_not_scanned=[(0, 1)], solvent_mol=None)

    assert len(ts_guess.active_bonds) == 2
    assert os.path.exists('h3_1_2_1dscan.png')

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(('.inp', '.png')) or 'animation' in filename:
            os.remove(filename)

    os.chdir(here)
