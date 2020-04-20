from autode.wrappers.ORCA import ORCA
from autode.calculation import Calculation
from autode.calculation import execute_calc
from autode.calculation import check_molecule_attr
from autode.molecule import Molecule
from autode.molecule import SolvatedMolecule
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoNormalModesFound
from autode.exceptions import NoInputError
from autode.exceptions import SolventUnavailable
from autode.solvent.solvents import Solvent
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
    assert calc.input_filename == 'opt_orca.inp'
    assert calc.output_filename == 'opt_orca.out'
    assert calc.terminated_normally is True
    assert calc.calculation_terminated_normally() is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False

    with pytest.raises(NoNormalModesFound):
        calc.get_normal_mode_displacements(mode_number=0)

    # Should have a partial atomic charge for every atom
    assert len(calc.get_atomic_charges()) == 5
    assert type(calc.get_atomic_charges()[0]) == float

    calc = Calculation(name='opt', molecule=methylchloride, method=method, opt=True,
                       keywords_list=['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'])

    # If the calculation is not run with calc.run() then there should be no input and the calc should
    # raise that there is no input
    with pytest.raises(NoInputError):
        execute_calc(calc)

    os.remove('opt_orca.inp')
    os.chdir(here)


def test_calc_bad_mol():

    class Mol:
        pass

    mol = Mol()

    with pytest.raises(AssertionError):
        Calculation(name='bad_mol_object', molecule=mol, method=method)

    mol.atoms = None
    mol.mult = 1
    mol.n_atoms = 0
    mol.charge = 0
    mol.solvent = None

    with pytest.raises(NoInputError):
        Calculation(name='no_atoms_mol', molecule=mol, method=method)

    solvent = Solvent(name='xx', smiles='X', aliases=['X'])
    mol.solvent = solvent
    with pytest.raises(SolventUnavailable):
        Calculation(name='no_atoms_mol', molecule=mol, method=method)


def test_orca_optts_calculation():
    # TODO check the number of atoms etc. matches between mol and the calculation output? i.e break and rewrite test

    os.chdir(os.path.join(here, 'data'))
    methane = SolvatedMolecule(name='methane', smiles='C')
    methane.qm_solvent_atoms = []

    calc = Calculation(name='test_ts_reopt_optts', molecule=methane, method=method, opt=True,
                       bond_ids_to_add=[(0, 1)],
                       keywords_list=['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                       other_input_block='%geom\nCalc_Hess true\nRecalc_Hess 40\nTrust 0.2\nMaxIter 100\nend')
    calc.run()

    assert os.path.exists('test_ts_reopt_optts_orca.inp')
    inp_lines = open('test_ts_reopt_optts_orca.inp', 'r').readlines()

    # If the 0, 1 is the active bond in the TS then all the atoms in methane are 'core'
    assert all(atom_index in calc.core_atoms for atom_index in [0, 1, 2, 3, 4])

    assert calc.get_normal_mode_displacements(mode_number=6) is not None
    assert calc.terminated_normally is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False
    assert len(calc.get_imag_freqs()) == 1

    # Gradients should be an n_atom x 3 array
    gradients = calc.get_gradients()
    assert len(gradients) == 5
    assert len(gradients[0]) == 3

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

    calc.output_file_lines = None
    assert calc.calculation_terminated_normally() is False


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
