from autode.transition_states import ts_guess
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.transition_state import TS
from autode.reactions import Substitution
from autode import molecule
from autode.transition_states import optts
from autode.config import Config
from autode.wrappers.ORCA import ORCA
import os
here = os.path.dirname(os.path.abspath(__file__))

Config.make_ts_template = False

test_ts_reac = molecule.Molecule(xyzs=[['F', -2.0, 0.0, 0.0],
                                       ['C', 0.0, 0.0, 0.0],
                                       ['Cl', 1.8, 0.0, 0.0],
                                       ['H', -0.4, -0.5, -0.9],
                                       ['H', -0.4, -0.5, 0.9],
                                       ['H', -0.5, 1.0, 0.0]])

test_ts_prod = molecule.Molecule(xyzs=[['F', -1.0, 0.0, 0.0],
                                       ['C', 0.0, 0.0, 0.0],
                                       ['Cl', 3.0, 0.0, 0.0],
                                       ['H', -0.4, -0.5, -0.9],
                                       ['H', -0.4, -0.5, 0.9],
                                       ['H', -0.5, 1.0, 0.0]])

test_ts_mol = molecule.Molecule(xyzs=[['F', -3.0, -0.1, 0.1],
                                      ['C', 0.0, 0.0, 0.0],
                                      ['Cl', 2.2, 0.1, -0.1],
                                      ['H', 0.0, -0.8, 0.7],
                                      ['H', -0.2, -0.3, -1.0],
                                      ['H', -0.2, 1.0, 0.3]],
                                solvent='water', charge=-1)

ts_guess_obj = ts_guess.TSguess(name='test_ts', molecule=test_ts_mol,
                                active_bonds=[(0, 1), (1, 2)], reaction_class=Substitution)


def test_ts_guess_class():
    os.chdir(os.path.join(here, 'data'))

    assert ts_guess_obj.optts_converged == False
    assert ts_guess_obj.optts_nearly_converged == False
    assert ts_guess_obj.xyzs == [['F', -3.0, -0.1, 0.1],
                                 ['C', 0.0, 0.0, 0.0],
                                 ['Cl', 2.2, 0.1, -0.1],
                                 ['H', 0.0, -0.8, 0.7],
                                 ['H', -0.2, -0.3, -1.0],
                                 ['H', -0.2, 1.0, 0.3]]

    ts_guess_obj.run_orca_optts()
    ts_guess_obj.check_optts_convergence()

    assert ts_guess_obj.optts_converged == False
    assert ts_guess_obj.optts_nearly_converged == True

    ts_guess_obj.check_optts_convergence()
    imag_freqs, ts_xyzs, ts_energy = ts_guess_obj.get_imag_frequencies_xyzs_energy()

    assert len(ts_guess_obj.xyzs) == 6
    assert imag_freqs == [-511.81]
    assert ts_energy == -599.478764991939
    assert ts_guess_obj.optts_converged == True

    # testing optts.get_displaced_xyzs_along_imaginary_mode
    displaced_xyzs = optts.get_displaced_xyzs_along_imaginary_mode(ts_guess_obj.optts_calc, 6)

    assert displaced_xyzs[0][1] == ts_xyzs[0][1] + 0.268035
    assert displaced_xyzs[3][2] == ts_xyzs[3][2] - 0.020503
    assert displaced_xyzs[5][3] == ts_xyzs[5][3] - 0.015893

    # testing optts.ts_has_correct_imaginary_vector
    assert optts.ts_has_correct_imaginary_vector(ts_guess_obj.optts_calc, 5, [(0, 1), (1, 2)]) == True
    assert optts.ts_has_correct_imaginary_vector(ts_guess_obj.optts_calc, 5, None) == True
    assert optts.ts_has_correct_imaginary_vector(ts_guess_obj.optts_calc, 5, [(3, 4)]) == False

    # testing optts.check_close_imag_contribution
    assert optts.check_close_imag_contribution(ts_guess_obj.optts_calc, (test_ts_reac, test_ts_prod), ORCA) == True

    # testing ts_guess.do_displacements
    ts_guess_obj.do_displacements(magnitude=1)
    assert ts_guess_obj.xyzs[0] == ['F', -2, 0, 0]

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)

    os.chdir(here)


def test_get_ts():
    os.chdir(os.path.join(here, 'data'))

    assert optts.get_ts(None) is None

    get_ts_output = optts.get_ts(ts_guess_obj)
    ts_obj = TS(get_ts_output[0], converged=get_ts_output[1])

    assert ts_obj.get_atom_label(0) == 'F'
    assert ts_obj.solvent == 'water'
    assert ts_obj.converged == True
    assert ts_obj.active_atoms == [0, 1, 2]
    assert ts_obj.graph.number_of_nodes() == 6
    assert ts_obj.graph.number_of_edges() == 5
    assert ts_obj.truncated_graph.number_of_nodes() == 6
    assert ts_obj.truncated_graph.number_of_edges() == 5
    assert ts_obj.is_true_ts() == True

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)

    os.chdir(here)


def test_ts_template():
    os.chdir(os.path.join(here, 'data'))

    get_ts_output = optts.get_ts(ts_guess_obj)
    ts_obj = TS(get_ts_output[0], converged=get_ts_output[1])

    ts_obj.save_ts_template(folder_path=here)
    assert len(get_ts_templates(reaction_class=ts_obj.reaction_class, folder_path=here)) >= 1
    assert os.path.exists(os.path.join(here, 'template0.obj'))
    os.remove(os.path.join(here, 'template0.obj'))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)
    os.chdir(here)
