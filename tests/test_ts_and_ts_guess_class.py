from autode.transition_states import ts_guess
from autode.transition_states.transition_state import TS
from autode import molecule
import os

test_ts_mol = molecule.Molecule(xyzs=[['F', -3.0, -0.1, 0.1],['C', 0.0, 0.0, 0.0],['Cl', 2.2, 0.1, -0.1],['H', 0.0, -0.8, 0.7],['H', -0.2, -0.3, -1.0],['H', -0.2, 1.0, 0.3]], solvent='water')
ts_guess_obj = ts_guess.TSguess(name='test_ts',molecule=test_ts_mol, active_bonds=[(0,1), (1,2)])


here = os.path.dirname(os.path.abspath(__file__))

def test_ts_guess_class():
    os.chdir(os.path.join(here, 'data'))

    assert ts_guess_obj.optts_converged == False
    assert ts_guess_obj.optts_nearly_converged == False
    assert ts_guess_obj.xyzs == [['F', -3.0, -0.1, 0.1],['C', 0.0, 0.0, 0.0],['Cl', 2.2, 0.1, -0.1],['H', 0.0, -0.8, 0.7],['H', -0.2, -0.3, -1.0],['H', -0.2, 1.0, 0.3]]

    ts_guess_obj.run_orca_optts()
    ts_guess_obj.check_optts_convergence()
    imag_freqs, ts_xyzs, ts_energy = ts_guess_obj.get_imag_frequencies_xyzs_energy()
    
    assert ts_guess_obj.optts_converged == False
    assert ts_guess_obj.optts_nearly_converged == True
    assert len(ts_guess_obj.xyzs) == 6
    assert imag_freqs == [-511.81]
    assert ts_energy == -599.478764991939

    ts_guess_obj.check_optts_convergence()
    
    assert ts_guess_obj.optts_converged == True

    os.remove('test_ts_optts_orca.inp')
    os.remove('test_ts_reopt_optts_orca.inp')
    os.chdir(here)


def test_ts_class():
    ts_obj = TS(ts_guess=ts_guess_obj, name='ts')
    assert ts_obj.solvent == 'water'
    assert ts_obj.converged == True
    assert ts_obj.active_atoms == [0,1,2]
    assert ts_obj.graph.number_of_nodes() == 6
    assert ts_obj.graph.number_of_edges() == 5
    assert ts_obj.truncated_graph.number_of_nodes() == 6
    assert ts_obj.truncated_graph.number_of_edges() == 5
    assert ts_obj.is_true_ts() == True