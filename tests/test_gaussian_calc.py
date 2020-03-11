from autode.wrappers.G09 import G09
from autode.calculation import Calculation
from autode.molecule import Molecule
from autode.exceptions import XYZsNotFound
from autode.exceptions import NoInputError
import pytest
import os


here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = G09()


def test_gauss_opt_calc():

    os.chdir(os.path.join(here, 'data'))

    methylchloride = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent_name='water')
    calc = Calculation(name='opt', molecule=methylchloride, method=method, opt=True,
                       keywords_list=['PBE1PBE/Def2SVP', 'Opt'])
    calc.run()

    assert os.path.exists('opt_g09.com') is True
    assert os.path.exists('opt_g09.log') is True
    assert len(calc.get_final_atoms()) == 5
    assert os.path.exists('opt_g09.xyz') is True
    assert calc.get_energy() == -499.729222331
    assert calc.output_file_exists is True
    assert calc.rev_output_file_lines is not None
    assert calc.output_file_lines is not None
    assert calc.get_imag_freqs() is None
    assert calc.get_normal_mode_displacements(mode_number=1) is None
    assert calc.input_filename == 'opt_g09.com'
    assert calc.output_filename == 'opt_g09.log'
    assert calc.terminated_normally is True
    assert calc.calculation_terminated_normally() is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False

    os.remove('opt_g09.com')
    os.chdir(here)


def test_gauss_optts_calc():

    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='test_ts_reopt_optts', molecule=test_mol, method=method, opt=True,
                       keywords_list=['PBE1PBE/Def2SVP', 'Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, NoTrustUpdate)', 'Freq'])
    calc.run()

    assert calc.get_normal_mode_displacements(mode_number=6) is not None
    assert calc.terminated_normally is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False
    assert len(calc.get_imag_freqs()) == 1

    os.remove('test_ts_reopt_optts_g09.com')
    os.chdir(here)


def test_bad_gauss_output():

    calc = Calculation(name='no_output', molecule=test_mol, method=method)
    calc.output_file_lines = []
    calc.rev_output_file_lines = []

    assert calc.get_energy() is None
    with pytest.raises(XYZsNotFound):
        calc.get_final_atoms()

    with pytest.raises(NoInputError):
        calc.execute_calculation()


def test_fix_angle_error():

    os.chdir(os.path.join(here, 'data'))

    mol = Molecule(smiles='CC/C=C/CO')

    calc = Calculation(name='angle_fail', molecule=mol, method=method, opt=True,
                       keywords_list=['PBE1PBE/Def2SVP', 'Opt'])
    calc.run()

    assert os.path.exists('angle_fail_cartesian_g09.com') is True
    assert os.path.exists('angle_fail_internal_g09.com') is True
    assert calc.output_filename == 'angle_fail_internal_g09.log'
    assert calc.terminated_normally is True

    os.remove('angle_fail_cartesian_g09.xyz')
    os.remove('angle_fail_g09.xyz')

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.com'):
            os.remove(filename)

    os.chdir(here)
