from autode.wrappers.ORCA import ORCA
from autode.atoms import Atom
from autode.calculation import Calculation
from autode.calculation import execute_calc
from autode.species.molecule import Molecule
from autode.species.molecule import SolvatedMolecule
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoNormalModesFound
from autode.exceptions import NoInputError
from autode.exceptions import SolventUnavailable
from autode.exceptions import UnsuppportedCalculationInput
from autode.wrappers.keywords import SinglePointKeywords, OptKeywords
from autode.solvent.solvents import Solvent
import numpy as np
import pytest

import os
here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = ORCA()
method.available = True

sp_keywords = SinglePointKeywords(['PBE', 'def2-SVP'])
opt_keywords = OptKeywords(['Opt', 'PBE', 'def2-SVP'])


def test_orca_opt_calculation():

    os.chdir(os.path.join(here, 'data'))

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
    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.get_imaginary_freqs() == []
    assert calc.input.filename == 'opt_orca.inp'
    assert calc.output.filename == 'opt_orca.out'
    assert calc.terminated_normally()

    assert calc.optimisation_converged()

    assert calc.optimisation_nearly_converged() is False

    with pytest.raises(NoNormalModesFound):
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
    with pytest.raises(NoInputError):
        execute_calc(calc)

    os.remove('opt_orca.inp')
    os.chdir(here)


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

    with pytest.raises(NoInputError):
        Calculation(name='no_atoms_mol', molecule=mol, method=method,
                    keywords=opt_keywords)

    mol = Molecule(name='methane', smiles='C')
    mol.solvent = Solvent(name='xx', smiles='X', aliases=['X'])

    with pytest.raises(SolventUnavailable):
        Calculation(name='no_atoms_mol', molecule=mol, method=method,
                    keywords=opt_keywords)


def test_orca_optts_calculation():

    os.chdir(os.path.join(here, 'data'))
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

    os.remove('test_ts_reopt_optts_orca.inp')
    os.chdir(here)


def test_bad_orca_output():

    calc = Calculation(name='no_output', molecule=test_mol, method=method,
                       keywords=opt_keywords)
    calc.output.file_lines = []
    calc.output.rev_file_lines = []

    assert calc.get_energy() is None
    with pytest.raises(AtomsNotFound):
        calc.get_final_atoms()

    with pytest.raises(NoInputError):
        calc.execute_calculation()

    calc.output_file_lines = None
    assert calc.terminated_normally() is False


def test_solvation():

    methane = Molecule(name='solvated_methane', smiles='C',
                       solvent_name='water')

    with pytest.raises(UnsuppportedCalculationInput):

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


def test_gradients():
    os.chdir(os.path.join(here, 'data', 'orca'))

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

    os.chdir(here)
