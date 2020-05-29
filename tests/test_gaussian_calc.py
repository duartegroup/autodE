from autode.wrappers.G09 import G09
from autode.calculation import Calculation
from autode.molecule import Molecule
from autode.wrappers.keywords import OptKeywords, SinglePointKeywords
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoInputError
from autode.exceptions import NoNormalModesFound
from autode.point_charges import PointCharge
import pytest
import os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = G09()
method.available = True

opt_keywords = OptKeywords(['PBE1PBE/Def2SVP', 'Opt'])
optts_keywords = OptKeywords(['PBE1PBE/Def2SVP', 'Freq',
                              'Opt=(TS, CalcFC, NoEigenTest, '
                              'MaxCycles=100, MaxStep=10, NoTrustUpdate)'])

sp_keywords = SinglePointKeywords(['PBE1PBE/Def2SVP'])


def test_gauss_opt_calc():

    os.chdir(os.path.join(here, 'data'))

    methylchloride = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent_name='water')
    calc = Calculation(name='opt', molecule=methylchloride, method=method,
                       keywords=opt_keywords)
    calc.run()

    assert os.path.exists('opt_g09.com')
    assert os.path.exists('opt_g09.log')
    assert len(calc.get_final_atoms()) == 5
    assert os.path.exists('opt_g09.xyz')
    assert calc.get_energy() == -499.729222331
    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.get_imaginary_freqs() == []

    with pytest.raises(NoNormalModesFound):
        calc.get_normal_mode_displacements(mode_number=1)

    assert calc.input.filename == 'opt_g09.com'
    assert calc.output.filename == 'opt_g09.log'
    assert calc.terminated_normally()
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False

    charges = calc.get_atomic_charges()
    assert len(charges) == methylchloride.n_atoms

    # Should be no very large atomic charges in this molecule
    assert all(-1.0 < c < 1.0 for c in charges)

    gradients = calc.get_gradients()
    assert len(gradients) == methylchloride.n_atoms
    assert len(gradients[0]) == 3

    # Should be no large forces for an optimised molecule
    assert sum(gradients[0]) < 0.1

    os.remove('opt_g09.com')
    os.chdir(here)


def test_gauss_optts_calc():

    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='test_ts_reopt_optts', molecule=test_mol,
                       method=method, keywords=optts_keywords,
                       bond_ids_to_add=[(0, 1)])
    calc.run()
    print(calc.input.added_internals)
    assert os.path.exists('test_ts_reopt_optts_g09.com')

    bond_added = False
    for line in open('test_ts_reopt_optts_g09.com', 'r'):
        if 'B' in line and len(line.split()) == 3:
            bond_added = True
            assert line.split()[0] == 'B'
            assert line.split()[1] == '1'
            assert line.split()[2] == '2'

    assert bond_added

    assert calc.get_normal_mode_displacements(mode_number=6) is not None
    assert calc.terminated_normally()
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False
    assert len(calc.get_imaginary_freqs()) == 1

    assert -40.324 < calc.get_free_energy() < -40.322
    assert -40.301 < calc.get_enthalpy() < -40.299

    os.remove('test_ts_reopt_optts_g09.com')
    os.chdir(here)


def test_bad_gauss_output():

    calc = Calculation(name='no_output', molecule=test_mol, method=method,
                       keywords=opt_keywords)
    calc.output_file_lines = []
    calc.rev_output_file_lines = []

    assert calc.get_energy() is None
    with pytest.raises(AtomsNotFound):
        calc.get_final_atoms()

    with pytest.raises(NoInputError):
        calc.execute_calculation()


def test_fix_angle_error():

    os.chdir(os.path.join(here, 'data'))

    mol = Molecule(smiles='CC/C=C/CO')

    calc = Calculation(name='angle_fail', molecule=mol, method=method,
                       keywords=opt_keywords)
    calc.run()

    assert os.path.exists('angle_fail_g09_cartesian.com') is True
    assert os.path.exists('angle_fail_g09_internal.com') is True
    assert calc.output.filename == 'angle_fail_g09_internal.log'
    assert calc.terminated_normally()

    os.remove('angle_fail_g09_cartesian.com')
    os.remove('angle_fail_g09.com')
    os.remove('angle_fail_g09_internal.com')
    os.chdir(here)


def test_constraints():
    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='const_dist_opt', molecule=test_mol, method=method,
                       keywords=opt_keywords, distance_constraints={(0, 1): 1.2})
    calc.run()
    opt_atoms = calc.get_final_atoms()

    assert 1.199 < np.linalg.norm(opt_atoms[0].coord - opt_atoms[1].coord) < 1.201

    calc = Calculation(name='const_cart_opt', molecule=test_mol, method=method,
                       keywords=opt_keywords, cartesian_constraints=[0])
    calc.run()
    opt_atoms = calc.get_final_atoms()
    assert np.linalg.norm(test_mol.atoms[0].coord - opt_atoms[0].coord) < 1E-3

    os.remove('const_cart_opt_g09.com')
    os.remove('const_dist_opt_g09.com')
    os.chdir(os.path.join(here))


def test_single_atom_opt():
    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='H', molecule=Molecule(smiles='[H]'), method=method,
                       keywords=opt_keywords, n_cores=2)
    calc.generate_input()
    assert os.path.exists('H_g09.com')

    input_file_lines = open('H_g09.com', 'r').readlines()

    n_cores_set = False
    for line in input_file_lines:
        if 'PBE' in line:
            assert 'Opt' not in line
        if '%nprocshared=2' in line:
            n_cores_set = True

    assert n_cores_set

    os.remove('H_g09.com')
    os.chdir(os.path.join(here))


def test_point_charge_calc():
    os.chdir(os.path.join(here, 'data'))
    # Methane single point using a point charge with a unit positive charge
    # located at (10, 10, 10)

    calc = Calculation(name='methane_point_charge', molecule=test_mol,
                       method=method,
                       keywords=sp_keywords,
                       point_charges=[PointCharge(charge=1.0, x=10.0, y=10.0, z=10.0)])
    calc.run()

    # Assert that the input file is in the expected configuration
    for line in open('methane_point_charge_g09.com', 'r'):
        if 'PBE' in line:
            assert 'Charge' in line

        if len(line.split()) == 4:
            if not line.split()[0].isdigit():
                continue

            x, y, z, charge = line.split()
            assert float(x) == 10.0
            assert float(y) == 10.0
            assert float(z) == 10.0
            assert float(charge) == 1.0

    assert -40.428 < calc.get_energy() < -40.427

    # Gaussian needs x-matrix and nosymm in the input line to run optimisations
    # with point charges..
    for opt_keyword in ['Opt', 'Opt=Tight', 'Opt=(Tight)']:
        calc = Calculation(name='methane_point_charge_o', molecule=test_mol,
                           method=method,
                           keywords=OptKeywords(['PBE1PBE/Def2SVP', opt_keyword]),
                           point_charges=[PointCharge(charge=1.0, x=3.0, y=3.0, z=3.0)])
        calc.generate_input()

        for line in open('methane_point_charge_o_g09.com', 'r').readlines():
            if 'PBE' in line:
                assert 'charge' in line.lower()
                assert 'z-matrix' in line.lower() and 'nosymm' in line.lower()
                break

    os.remove('methane_point_charge_g09.com')
    os.remove('methane_point_charge_o_g09.com')
    os.chdir(os.path.join(here))
