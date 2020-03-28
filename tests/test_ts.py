from autode.transition_states import ts_guess
from autode.atoms import Atom
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.ts_guess import TSguess
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.transition_state import TransitionState
from autode.reactions import Substitution
from autode import molecule
from autode.molecule import Reactant, Product
from autode.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.wrappers.ORCA import ORCA
from autode.solvent.solvents import Solvent
import os
here = os.path.dirname(os.path.abspath(__file__))

Config.make_ts_template = False

ch3cl = Reactant(charge=0, mult=1, atoms=[Atom('Cl', 1.63664, 0.02010, -0.05829),
                                          Atom('C', -0.14524, -0.00136, 0.00498),
                                          Atom('H', -0.52169, -0.54637, -0.86809),
                                          Atom('H', -0.45804, -0.50420, 0.92747),
                                          Atom('H', -0.51166, 1.03181, -0.00597)])

f = Reactant(charge=-1, mult=1, atoms=[Atom('F', 4.0, 0.0, 0.0)])

ch3f = Product(charge=0, mult=1, atoms=[Atom('C', -0.05250, 0.00047, -0.00636),
                                        Atom('F', 1.31229, -0.01702, 0.16350),
                                        Atom('H', -0.54993, -0.04452, 0.97526),
                                        Atom('H', -0.34815, 0.92748, -0.52199),
                                        Atom('H', -0.36172, -0.86651, -0.61030)])

cl = Reactant(charge=-1, mult=1, atoms=[Atom('Cl', 4.0, 0.0, 0.0)])

f_ch3cl_tsguess = TSguess(reactant=ReactantComplex(f, ch3cl),
                          product=ProductComplex(ch3f, cl),
                          atoms=[Atom('F', -2.66092, -0.01426, 0.09700),
                                 Atom('Cl', 1.46795, 0.05788, -0.06166),
                                 Atom('C', -0.66317, -0.01826, 0.02488),
                                 Atom('H', -0.78315, -0.58679, -0.88975),
                                 Atom('H', -0.70611, -0.54149, 0.97313),
                                 Atom('H', -0.80305, 1.05409, 0.00503)])

f_ch3cl_ts = TransitionState(ts_guess=f_ch3cl_tsguess,
                             bond_rearrangement=BondRearrangement(breaking_bonds=[(2, 1)],
                                                                  forming_bonds=[(0, 2)]))


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

    ts_guess_obj.run_optts()
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
    displaced_xyzs = optts.get_displaced_xyzs_along_imaginary_mode(ts_guess_obj.optts_calc, 6, 6)

    assert displaced_xyzs[0][1] == ts_xyzs[0][1] + 0.268035
    assert displaced_xyzs[3][2] == ts_xyzs[3][2] - 0.020503
    assert displaced_xyzs[5][3] == ts_xyzs[5][3] - 0.015893

    # testing optts.ts_has_correct_imaginary_vector
    assert optts.ts_has_correct_imaginary_vector(ts_guess_obj.optts_calc, 5, [(0, 1), (1, 2)]) == True
    assert optts.ts_has_correct_imaginary_vector(ts_guess_obj.optts_calc, 5, None) == True
    assert optts.ts_has_correct_imaginary_vector(ts_guess_obj.optts_calc, 5, [(3, 4)]) == False

    # testing optts.check_close_imag_contribution
    assert optts.check_close_imag_contribution(ts_guess_obj.optts_calc, 6, (test_ts_reac, test_ts_prod), method) == True

    # testing ts_guess.do_displacements
    ts_guess_obj.do_displacements(size=1)
    assert ts_guess_obj.xyzs[0] == ['F', -2, 0, 0]

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)

    os.chdir(here)


def test_get_ts():
    os.chdir(os.path.join(here, 'data'))

    assert optts.get_ts(None)[0] is None

    get_ts_output = optts.get_ts(ts_guess_obj)
    ts_obj = TransitionState(get_ts_output[0], converged=get_ts_output[1])

    assert ts_obj.get_atom_label(0) == 'F'
    assert type(ts_obj.solvent) == Solvent
    assert ts_obj.solvent.name == 'water'
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
    ts_obj = TransitionState(get_ts_output[0], converged=get_ts_output[1])

    ts_obj.save_ts_template(folder_path=here)
    assert len(get_ts_templates(reaction_type=ts_obj.reaction_class, folder_path=here)) >= 1
    assert os.path.exists(os.path.join(here, 'template0.obj'))
    os.remove(os.path.join(here, 'template0.obj'))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)
    os.chdir(here)
