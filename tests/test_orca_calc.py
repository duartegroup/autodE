from autode.wrappers.ORCA import ORCA
from autode.calculation import Calculation
from autode.molecule import Molecule
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoInputError
import pytest

import os
here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = ORCA()


def test_orca_opt_calculation():

    os.chdir(os.path.join(here, 'data'))

    methylchloride = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent_name='water')
    calc = Calculation(name='opt', molecule=methylchloride, method=method, opt=True,
                       keywords_list=['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'])
    calc.run()

    assert os.path.exists('opt_orca.inp') is True
    assert os.path.exists('opt_orca.out') is True
    assert len(calc.get_final_atoms()) == 5
    assert calc.get_energy() == -499.730559818596
    assert calc.output_file_exists is True
    assert calc.rev_output_file_lines is not None
    assert calc.output_file_lines is not None
    assert calc.get_imag_freqs() == []
    assert calc.get_normal_mode_displacements(mode_number=1) is None
    assert calc.input_filename == 'opt_orca.inp'
    assert calc.output_filename == 'opt_orca.out'
    assert calc.terminated_normally is True
    assert calc.calculation_terminated_normally() is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False

    os.remove('opt_orca.inp')
    os.chdir(here)


def test_orca_optts_calculation():
    # TODO check the number of atoms etc. matches between mol and the calculation output? i.e break and rewrite test

    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='test_ts_reopt_optts', molecule=test_mol, method=method, opt=True,
                       keywords_list=['Opt', 'PBE0', 'RIJCOSX',
                                 'D3BJ', 'def2-SVP', 'def2/J'],
                       other_input_block='%geom\nCalc_Hess true\nRecalc_Hess 40\nTrust 0.2\nMaxIter 100\nend')
    calc.run()

    assert calc.get_normal_mode_displacements(mode_number=6) is not None
    assert calc.terminated_normally is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False
    assert len(calc.get_imag_freqs()) == 1

    os.remove('test_ts_reopt_optts_orca.inp')
    os.chdir(here)


def test_bad_orca_output():

    calc = Calculation(name='no_output', molecule=test_mol, method=method)
    calc.output_file_lines = []
    calc.rev_output_file_lines = []

    assert calc.get_energy() is None
    with pytest.raises(AtomsNotFound):
        calc.get_final_atoms()

    with pytest.raises(NoInputError):
        calc.execute_calculation()


def test_subprocess_to_output():

    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='test', molecule=test_mol, method=method)

    # Can't execute orca in the CI environment so check at least the subprocess works
    calc.input_filename = 'test_subprocess.py'
    with open(calc.input_filename, 'w') as test_file:
        print("print('hello world')", file=test_file)
    calc.output_filename = 'test_subprocess.out'

    # Overwrite the orca path
    calc.method.path = 'python'
    calc.execute_calculation()

    assert calc.output_file_lines == ['hello world\n']
    assert len(calc.output_file_lines) == 1
    assert len(calc.rev_output_file_lines) == 1

    os.remove('test_subprocess.out')
    os.chdir(here)
