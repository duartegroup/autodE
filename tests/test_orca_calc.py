import autode.exceptions as ex
from autode.wrappers.ORCA import ORCA
from autode.atoms import Atom
from autode.constants import Constants
from autode.wrappers.ORCA import calc_atom_entropy
from autode.calculation import Calculation, CalculationOutput
from autode.calculation import execute_calc
from autode.species.molecule import Molecule
from autode.species.molecule import SolvatedMolecule
from autode.input_output import xyz_file_to_atoms
from autode.wrappers.keywords import SinglePointKeywords, OptKeywords
from autode.wrappers.keywords import Functional, WFMethod, BasisSet
from autode.solvent.solvents import Solvent
from autode.config import Config
from autode import utils
from . import testutils
import numpy as np
import pytest

import os
here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = ORCA()

sp_keywords = SinglePointKeywords(['PBE', 'def2-SVP'])
opt_keywords = OptKeywords(['Opt', 'PBE', 'def2-SVP'])


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'orca.zip'))
def test_orca_opt_calculation():

    methylchloride = Molecule(name='CH3Cl',
                              smiles='[H]C([H])(Cl)[H]',
                              solvent_name='water')

    calc = Calculation(name='opt', molecule=methylchloride, method=method,
                       keywords=opt_keywords)
    calc.run()

    assert os.path.exists('opt_orca.inp') is True
    assert os.path.exists('opt_orca.out') is True
    assert len(calc.get_final_atoms()) == 5
    assert -499.735 < calc.get_energy() < -499.730
    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.get_imaginary_freqs() == []
    assert calc.input.filename == 'opt_orca.inp'
    assert calc.output.filename == 'opt_orca.out'
    assert calc.terminated_normally()

    assert calc.optimisation_converged()

    assert calc.optimisation_nearly_converged() is False

    with pytest.raises(ex.NoNormalModesFound):
        calc.get_normal_mode_displacements(mode_number=0)

    # Should have a partial atomic charge for every atom
    charges = calc.get_atomic_charges()
    assert len(charges) == 5
    assert type(charges[0]) == float
    assert -1.0 < charges[0] < 1.0

    calc = Calculation(name='opt', molecule=methylchloride, method=method,
                       keywords=opt_keywords)

    # If the calculation is not run with calc.run() then there should be no
    # input and the calc should raise that there is no input
    with pytest.raises(ex.NoInputError):
        execute_calc(calc)


def test_calc_bad_mol():

    class Mol:
        pass

    mol = Mol()

    with pytest.raises(AssertionError):
        Calculation(name='bad_mol_object', molecule=mol, method=method,
                    keywords=opt_keywords)

    mol.atoms = None
    mol.mult = 1
    mol.n_atoms = 0
    mol.charge = 0
    mol.solvent = None

    with pytest.raises(ex.NoInputError):
        Calculation(name='no_atoms_mol', molecule=mol, method=method,
                    keywords=opt_keywords)

    mol = Molecule(name='methane', smiles='C')
    mol.solvent = Solvent(name='xx', smiles='X', aliases=['X'])

    with pytest.raises(ex.SolventUnavailable):
        Calculation(name='no_atoms_mol', molecule=mol, method=method,
                    keywords=opt_keywords)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'orca.zip'))
def test_orca_optts_calculation():

    methane = SolvatedMolecule(name='methane', smiles='C')
    methane.qm_solvent_atoms = []

    calc = Calculation(name='test_ts_reopt_optts', molecule=methane,
                       method=method,
                       bond_ids_to_add=[(0, 1)],
                       keywords=opt_keywords,
                       other_input_block='%geom\n'
                                         'Calc_Hess true\n'
                                         'Recalc_Hess 40\n'
                                         'Trust 0.2\n'
                                         'MaxIter 100\nend')
    calc.run()

    assert os.path.exists('test_ts_reopt_optts_orca.inp')

    assert calc.get_normal_mode_displacements(mode_number=6) is not None
    assert calc.terminated_normally()
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False
    assert len(calc.get_imaginary_freqs()) == 1

    # Gradients should be an n_atom x 3 array
    gradients = calc.get_gradients()
    assert len(gradients) == 5
    assert len(gradients[0]) == 3

    assert -599.437 < calc.get_enthalpy() < -599.436
    assert -599.469 < calc.get_free_energy() < -599.468


def test_bad_orca_output():

    calc = Calculation(name='no_output', molecule=test_mol, method=method,
                       keywords=opt_keywords)

    assert calc.get_energy() is None
    with pytest.raises(ex.AtomsNotFound):
        calc.get_final_atoms()

    with pytest.raises(ex.NoInputError):
        calc.execute_calculation()

    calc.output_file_lines = None
    assert calc.terminated_normally() is False


def test_solvation():

    methane = Molecule(name='solvated_methane', smiles='C',
                       solvent_name='water')

    with pytest.raises(ex.UnsuppportedCalculationInput):

        # Should raise on unsupported calculation type
        method.implicit_solvation_type = 'xxx'
        calc = Calculation(name='broken_solvation', molecule=methane,
                           method=method, keywords=sp_keywords)
        calc.run()

    method.implicit_solvation_type = 'CPCM'
    calc = Calculation(name='methane_cpcm', molecule=methane,
                       method=method, keywords=sp_keywords)
    calc.generate_input()

    assert any('cpcm' in line.lower() for line in open('methane_cpcm_orca.inp', 'r'))
    os.remove('methane_cpcm_orca.inp')

    method.implicit_solvation_type = 'SMD'
    calc = Calculation(name='methane_smd', molecule=methane,
                       method=method, keywords=sp_keywords)
    calc.generate_input()

    assert any('smd' in line.lower() for line in open('methane_smd_orca.inp', 'r'))
    os.remove('methane_smd_orca.inp')


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'orca.zip'))
def test_gradients():

    h2 = Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.0)])
    calc = Calculation(name='h2_grad', molecule=h2,
                       method=method,
                       keywords=method.keywords.grad)
    calc.run()
    h2.energy = calc.get_energy()

    delta_r = 1E-8

    # Energy of a finite difference approximation
    h2_disp = Molecule(name='h2_disp',
                       atoms=[Atom('H'), Atom('H', x=1.0 + delta_r)])
    calc = Calculation(name='h2_disp', molecule=h2_disp,
                       method=method,
                       keywords=method.keywords.grad)
    calc.run()
    h2_disp.energy = calc.get_energy()

    delta_energy = h2_disp.energy - h2.energy   # Ha
    grad = delta_energy / delta_r               # Ha A^-1

    calc = Calculation(name='h2_grad', molecule=h2,
                            method=method,
                            keywords=method.keywords.grad)

    calc.run()

    diff = calc.get_gradients()[1, 0] - grad    # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1E-3


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'orca.zip'))
def test_mp2_numerical_gradients():

    calc = Calculation(name='tmp',
                       molecule=Molecule(atoms=xyz_file_to_atoms('tmp_orca.xyz')),
                       method=method,
                       keywords=method.keywords.grad)
    calc.output = CalculationOutput(filename='tmp_orca.out')

    gradients = calc.get_gradients()
    assert len(gradients) == 6
    expected = np.array([-0.00971201,  -0.00773534,  -0.02473580]) / Constants.a0_to_ang
    assert np.linalg.norm(expected - gradients[0]) < 1e-6

    # Test for different printing with numerical..
    calc.output = CalculationOutput(filename='numerical_orca.out')

    gradients = calc.get_gradients()
    assert len(gradients) == 6
    expected = np.array([0.012397372, 0.071726232, -0.070942743]) / Constants.a0_to_ang
    assert np.linalg.norm(expected - gradients[0]) < 1e-6


def test_calc_entropy():

    f_entropy_g09 = 0.011799 / 298.15   # TS from g09
    f_entropy = calc_atom_entropy(atom_label='F', temp=298.15)

    # Ensure the calculated and 'actual' from Gaussian09 are close
    assert np.abs(f_entropy_g09 - f_entropy) < 2E-5


@utils.work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_keyword_setting():

    orca = ORCA()
    orca.keywords.sp.functional = 'B3LYP'

    # Setter should generate a Functional from the keyword string
    assert isinstance(orca.keywords.sp.functional, Functional)

    calc = Calculation(name='tmp',
                       molecule=test_mol.copy(),
                       method=orca,
                       keywords=orca.keywords.sp)
    calc.generate_input()
    assert calc.input.exists

    # B3LYP should now be in the in input
    inp_lines = open(calc.input.filename, 'r').readlines()
    assert any('B3LYP' in line for line in inp_lines)

    # With a keyword without ORCA defined then raise an exception
    with pytest.raises(ex.UnsuppportedCalculationInput):
        orca.keywords.sp.functional = Functional(name='B3LYP', g09='B3LYP')
        calc = Calculation(name='tmp',
                           molecule=test_mol.copy(),
                           method=orca,
                           keywords=orca.keywords.sp)
        calc.generate_input()

    # Without a default wavefunction method defined in the single point method
    # we can't set keywords.wf
    with pytest.raises(ValueError):
        orca.keywords.sp.wf_method = 'HF'

    # but if we have a WF method in the keywords we should be able to set it
    orca.keywords.sp = SinglePointKeywords([WFMethod('MP2'),
                                            BasisSet('def2-TZVP')])

    orca.keywords.sp.wf_method = 'HF'
    assert orca.keywords.sp.wf_method == 'HF'


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'orca.zip'))
def test_hessian_extraction():

    calc = Calculation(name='tmp',
                       molecule=Molecule(smiles='O'),
                       method=method,
                       keywords=method.keywords.hess)

    calc.output.filename = 'H2O_hess_orca.out'

    with open('H2O_hess_orca.xyz', 'w') as xyz_file:
        print('3\n',
              'O     -0.001100    0.363100   -0.000000',
              'H     -0.825000   -0.181900   -0.000000',
              'H      0.826100   -0.181200    0.000000',
              sep='\n',
              file=xyz_file)

    hessian = calc.get_hessian()
    assert hessian.shape == (9, 9)
    # should not have any very large values
    assert np.sum(np.abs(hessian)) < 100

    calc.output = CalculationOutput(filename='no_file.out')
    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_hessian()

    calc.output = CalculationOutput(filename='H2O_hess_broken.out')
    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_hessian()
