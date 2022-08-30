from autode.calculations import Calculation, CalculationOutput
from autode.calculations.executors import CalculationExecutor
from autode.solvent.solvents import get_solvent
from autode.constraints import Constraints
from autode.wrappers.keywords import SinglePointKeywords
from autode.wrappers.keywords.functionals import Functional
from autode.utils import run_external
from autode.methods import XTB, ORCA
from autode.species import Molecule
from autode.config import Config
import autode.exceptions as ex
from autode.utils import work_in_tmp_dir
from .testutils import requires_with_working_xtb_install
from copy import deepcopy
import pytest
import os
import sys

test_mol = Molecule(smiles='O', name='test_mol')


@work_in_tmp_dir()
def test_calc_class():

    xtb = XTB()

    calc = Calculation(name='-tmp',
                       molecule=test_mol,
                       method=xtb,
                       keywords=xtb.keywords.sp)

    # Should prepend a dash to appease some EST methods
    assert not calc._executor.name.startswith('-')
    assert calc.molecule is not None
    assert calc.method.name == 'xtb'
    assert len(calc.input.filenames) == 0

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_energy()

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_final_atoms()

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_gradients()

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_atomic_charges()

    # With a filename that doesn't exist a NoOutput exception should be raised
    calc.output.filename = '/a/path/that/does/not/exist/tmp'
    with pytest.raises(ex.NoCalculationOutput):
        _ = calc.output.file_lines

    # With no output should not be able to get properties
    calc.output.filename = 'tmp.out'
    with open(calc.output.filename, 'w') as output_file:
        print('some\ntest\noutput', file=output_file)

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_atomic_charges()

    # Should default to a single core
    assert calc.n_cores == 1

    calc_str = str(calc)
    new_calc = Calculation(name='tmp2',
                           molecule=test_mol,
                           method=xtb,
                           keywords=xtb.keywords.sp)
    new_calc_str = str(new_calc)
    # Calculation strings need to be unique
    assert new_calc_str != calc_str

    new_calc = Calculation(name='tmp2',
                           molecule=test_mol,
                           method=xtb,
                           keywords=xtb.keywords.opt)
    assert str(new_calc) != new_calc_str

    mol_no_atoms = Molecule()
    with pytest.raises(ex.NoInputError):
        _ = Calculation(name='tmp2',
                        molecule=mol_no_atoms,
                        method=xtb,
                        keywords=xtb.keywords.sp)


def test_calc_copy():

    orca = ORCA()
    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=orca,
                       keywords=orca.keywords.sp)

    copied_calc = calc.copy()
    copied_calc.input.keywords = None

    assert calc.input.keywords is not None


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_clear_output():

    with open('tmp.out', 'w') as out_file:
        print('some', 'test', 'output', sep='\n', file=out_file)

    output = CalculationOutput(filename='tmp.out')
    assert output.exists

    assert len(output.file_lines) == 3

    with open('tmp.out', 'w') as out_file:
        print('new output', sep='\n', file=out_file)

    # Without clearing the output then the file lines are not updated
    assert len(output.file_lines) == 3

    # Clearing the output will clear the cached property (file_lines). Lines
    # are reloaded when the file_lines property is accessed again
    output.clear()

    assert output.exists
    assert len(output.file_lines) == 1

    os.remove('tmp.out')


def test_distance_const_check():

    # Cannot have distance constraints between identical atoms
    assert Constraints(distance={(0, 0): 0.0},
                       cartesian=None).distance is None

    assert len(Constraints(distance={(0, 0): 0.0, (1, 0): 1.0},
                           cartesian=None).distance) == 1


def test_calc_string():

    xtb = XTB()

    a = test_mol.copy()
    no_const = Calculation(name='tmp',
                           molecule=a,
                           method=xtb,
                           keywords=xtb.keywords.sp)

    b = test_mol.copy()
    b.constraints.cartesian = [0]
    cart_const = Calculation(name='tmp',
                             molecule=b,
                             method=xtb,
                             keywords=xtb.keywords.sp)

    c = test_mol.copy()
    c.constraints.distance = {(0, 1): 1.0}
    dist_const = Calculation(name='tmp',
                             molecule=c,
                             method=xtb,
                             keywords=xtb.keywords.sp)

    d = test_mol.copy()
    d.constraints.distance = {(0, 1): 1.5}
    dist_const2 = Calculation(name='tmp',
                              molecule=d,
                              method=xtb,
                              keywords=xtb.keywords.sp)

    assert str(no_const) == str(no_const)
    assert str(no_const) != str(cart_const)
    assert str(no_const) != str(dist_const)
    assert str(cart_const) != str(dist_const)
    assert str(dist_const) != str(dist_const2)


@work_in_tmp_dir()
def test_fix_unique():
    """So calculations with different input but the same name are not skipped
    autodE checks the input of each previously run calc with the name name"""

    orca = ORCA()

    calc = CalculationExecutor(name='tmp',
                               molecule=test_mol,
                               method=orca,
                               keywords=orca.keywords.sp)
    calc._fix_unique()
    assert calc.name == 'tmp_orca'

    # Should generate a register
    assert os.path.exists('.autode_calculations')
    assert len(open('.autode_calculations', 'r').readlines()) == 1

    calc = CalculationExecutor(name='tmp',
                               molecule=test_mol,
                               method=orca,
                               keywords=orca.keywords.opt)
    calc._fix_unique()
    assert calc.name != 'tmp_orca'
    assert calc.name == 'tmp_orca0'

    # no need to fix unique if the name is different
    calc = CalculationExecutor(name='tmp2',
                               molecule=test_mol,
                               method=orca,
                               keywords=orca.keywords.opt)
    calc._fix_unique()
    assert calc.name == 'tmp2_orca'


def test_solvent_get():
    xtb = XTB()
    _test_mol = Molecule(smiles='O', name='test_mol')

    # Can't get the name of a solvent if molecule.solvent is not a string
    with pytest.raises(ex.SolventUnavailable):
        _test_mol.solvent = 5

    with pytest.raises(ex.SolventNotFound):
        _test_mol.solvent = 'a_solvent_that_doesnt_exist'

    # Should work fine with a normal solvent
    _test_mol.solvent = get_solvent('water', kind='implicit')
    assert _test_mol.solvent.xtb.lower() in ['water', 'h2o']

    # Currently iodoethane is not in XTB - might be in the future
    _test_mol.solvent = 'iodoethane'
    assert not hasattr(test_mol.solvent, 'xtb')
    assert _test_mol.solvent.is_implicit

    with pytest.raises(ex.SolventUnavailable):
        _ = Calculation('test',
                        molecule=_test_mol,
                        method=xtb,
                        keywords=xtb.keywords.sp)


@work_in_tmp_dir()
def test_input_gen():

    xtb = XTB()

    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=xtb,
                       keywords=xtb.keywords.sp)

    Config.keep_input_files = True
    calc.generate_input()
    assert os.path.exists('tmp_xtb.xyz')
    calc.clean_up()
    # Clean-up should do nothing if keep_input_files = True
    assert os.path.exists('tmp_xtb.xyz')

    # but should be able to be forced
    calc.clean_up(force=True)
    assert not os.path.exists('tmp_xtb.xyz')

    # Test the keywords parsing
    unsupported_func = Functional('PBE', orca='PBE')
    calc_kwds = Calculation(name='tmp',
                            molecule=test_mol,
                            method=xtb,
                            keywords=SinglePointKeywords([unsupported_func]))

    with pytest.raises(ex.UnsupportedCalculationInput):
        calc_kwds.generate_input()


@work_in_tmp_dir()
def test_exec_not_avail_method():

    orca = ORCA()
    orca.path = '/a/non/existent/path'
    assert not orca.is_available

    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=orca,
                       keywords=orca.keywords.sp)
    calc.generate_input()

    with pytest.raises(ex.MethodUnavailable):
        calc._executor.run()

    with pytest.raises(ex.MethodUnavailable):
        calc.run()


@work_in_tmp_dir()
def test_exec_too_much_memory_requested_py39():

    if sys.version_info.minor != 9:
        return  # Only supported on Python 3.9

    # Normal external run should be fine
    run_external(['ls'], output_filename='tmp.txt')

    curr_max_core = deepcopy(Config.max_core)
    Config.max_core = 10000000000000

    # But if there is not enough physical memory it should raise an exception
    with pytest.raises(RuntimeError):
        run_external(['ls'], output_filename='tmp.txt')

    Config.max_core = curr_max_core


@work_in_tmp_dir()
def test_exec_too_much_memory_requested_py38():

    if sys.version_info.minor != 8:
        return  # Only supported on Python 3.8

    # Python 3.8 can't use os.sysconf, so check that external can still be run
    run_external(['ls'], output_filename='tmp.txt')


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_calculations_have_unique_names():

    xtb = XTB()
    mol = Molecule(smiles='O')

    mol.single_point(method=xtb)
    mol.single_point(method=xtb)  # calculation should be skipped

    """For some insane reason the following code works if executed in python 
    directly but not if run within pytest"""
    # neutral_energy = mol.energy.copy()
    #
    # mol.charge = 1
    # mol.single_point(method=xtb)  # Calculation should be rerun
    # cation_energy = mol.energy
    # assert cation_energy > neutral_energy
