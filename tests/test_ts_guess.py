from autode.transition_states import ts_guess
import os
from autode import molecule
from autode.transition_states import optts
from autode.calculation import Calculation
from autode.wrappers.ORCA import ORCA
from autode import Config

test_ts = molecule.Molecule(xyzs=[['F', -3.0, -0.1, 0.1],['C', 0.0, 0.0, 0.0],['Cl', 2.2, 0.1, -0.1],['H', 0.0, -0.8, 0.7],['H', -0.2, -0.3, -1.0],['H', -0.2, 1.0, 0.3]])
ts_guess_obj = ts_guess.TSguess(name='test_ts',molecule=test_ts, active_bonds=[(0,1), (1,2)])


here = os.path.dirname(os.path.abspath(__file__))

def test_ts_guess():
    os.chdir(os.path.join(here, 'data'))

    assert ts_guess_obj.optts_converged == False
    assert ts_guess_obj.xyzs == [['F', -3.0, -0.1, 0.1],['C', 0.0, 0.0, 0.0],['Cl', 2.2, 0.1, -0.1],['H', 0.0, -0.8, 0.7],['H', -0.2, -0.3, -1.0],['H', -0.2, 1.0, 0.3]]

    ts_guess_obj.run_orca_optts()
    ts_guess_obj.check_optts_convergence()
    imag_freqs, ts_xyzs, ts_energy = ts_guess_obj.get_imag_frequencies_xyzs_energy()
    
    assert ts_guess_obj.optts_converged == True
    assert len(ts_guess_obj.xyzs) == 6
    assert imag_freqs == [-511.81]
    assert ts_energy == -599.478764991939

    os.remove('test_ts_optts_orca.inp')
    os.chdir(here)
