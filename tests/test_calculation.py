from autode.calculation import Calculation, Constraints, CalculationOutput
from autode.solvent.solvents import get_solvent
from autode.wrappers.keywords import SinglePointKeywords
from autode.wrappers.functionals import Functional
from autode.methods import XTB, ORCA
from autode.species import Molecule
from autode.config import Config
import autode.exceptions as ex
from autode.utils import work_in_tmp_dir
import pytest
import os

test_mol = Molecule(smiles='O', name='test_mol')


def test_calc_class():

    xtb = XTB()

    calc = Calculation(name='-tmp',
                       molecule=test_mol,
                       method=xtb,
                       keywords=xtb.keywords.sp)

    # Should prepend a dash to appease some EST methods
    assert not calc.name.startswith('-')
    assert calc.molecule is not None
    assert calc.method.name == 'xtb'
    assert len(calc.input.filenames) == 0

    assert calc.get_energy() is None

    assert not calc.optimisation_converged()
    assert not calc.optimisation_nearly_converged()

    with pytest.raises(ex.AtomsNotFound):
        _ = calc.get_final_atoms()

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_gradients()

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_atomic_charges()

    # Calculation that has not been run shouldn't have an opt converged
    assert not calc.optimisation_converged()
    assert not calc.optimisation_nearly_converged()

    # With a filename that doesn't exist a NoOutput exception should be raised
    calc.output.filename = '/a/path/that/does/not/exist/tmp'
    with pytest.raises(ex.NoCalculationOutput):
        _ = calc.output.file_lines

    # With no output should not be able to get properties
    calc.output.filename = 'tmp.out'
    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_atomic_charges()

    # or final atoms
    with pytest.raises(ex.AtomsNotFound):
        _ = calc.get_final_atoms()

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

    # Clearing the output will clear the cached property (file_lines) and
    # reload the new lines
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

    no_const = Calculation(name='tmp',
                           molecule=test_mol,
                           method=xtb,
                           keywords=xtb.keywords.sp)

    cart_const = Calculation(name='tmp',
                             molecule=test_mol,
                             method=xtb,
                             keywords=xtb.keywords.sp,
                             cartesian_constraints=[0])

    dist_const = Calculation(name='tmp',
                             molecule=test_mol,
                             method=xtb,
                             keywords=xtb.keywords.sp,
                             distance_constraints={(0, 1): 1.0})

    dist_const2 = Calculation(name='tmp',
                              molecule=test_mol,
                              method=xtb,
                              keywords=xtb.keywords.sp,
                              distance_constraints={(0, 1): 1.5})

    assert str(no_const) == str(no_const)
    assert str(no_const) != str(cart_const)
    assert str(no_const) != str(dist_const)
    assert str(cart_const) != str(dist_const)
    assert str(dist_const) != str(dist_const2)


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_fix_unique():
    """So calculations with different input but the same name are not skipped
    autodE checks the input of each previously run calc with the name name"""

    orca = ORCA()

    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=orca,
                       keywords=orca.keywords.sp)
    calc._fix_unique()
    assert calc.name == 'tmp_orca'

    # Should generate a register
    assert os.path.exists('.autode_calculations')
    assert len(open('.autode_calculations', 'r').readlines()) == 1

    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=orca,
                       keywords=orca.keywords.opt)
    calc._fix_unique()
    assert calc.name != 'tmp_orca'
    assert calc.name == 'tmp_orca0'

    # no need to fix unique if the name is different
    calc = Calculation(name='tmp2',
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
                        method=xtb, keywords=xtb.keywords.sp)


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
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

    with pytest.raises(ex.UnsuppportedCalculationInput):
        calc_kwds.generate_input()


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_exec_not_avail_method():

    orca = ORCA()
    orca.path = '/a/non/existent/path'
    assert not orca.available

    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=orca,
                       keywords=orca.keywords.sp)
    calc.generate_input()

    with pytest.raises(ex.MethodUnavailable):
        calc.execute_calculation()

    with pytest.raises(ex.MethodUnavailable):
        calc.run()
