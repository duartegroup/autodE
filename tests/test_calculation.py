from autode.calculation import Calculation, get_solvent_name
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

    assert calc.get_energy() is None
    assert calc.get_enthalpy() is None
    assert calc.get_free_energy() is None

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
        calc.output.set_lines()

    # With no output should not be able to get properties
    calc.output.filename = 'tmp'
    calc.output.file_lines = []
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
                           keywords=xtb.keywords.sp,
                           temp=5000)
    assert str(new_calc) != new_calc_str

    mol_no_atoms = Molecule()
    with pytest.raises(ex.NoInputError):
        _ = Calculation(name='tmp2',
                        molecule=mol_no_atoms,
                        method=xtb,
                        keywords=xtb.keywords.sp)


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

    # Can't get the name of a solvent if molecule.solvent is not a string
    test_mol.solvent = 5
    with pytest.raises(ex.SolventUnavailable):
        _ = get_solvent_name(molecule=test_mol,  method=xtb)

    test_mol.solvent = None
    assert get_solvent_name(test_mol, method=xtb) is None

    test_mol.solvent = 'a_solvent_that_doesnt_exist'
    with pytest.raises(ex.SolventNotFound):
        _ = get_solvent_name(molecule=test_mol,  method=xtb)

    # Should work fine with a normal solvent
    test_mol.solvent = get_solvent(solvent_name='water')
    solv_name_xtb = get_solvent_name(test_mol, method=xtb)
    assert solv_name_xtb.lower() in ['water', 'h2o']

    # Currently iodoethane is not in XTB - might be in the future
    test_mol.solvent = get_solvent(solvent_name='iodoethane')
    with pytest.raises(ex.SolventUnavailable):
        _ = get_solvent_name(test_mol, method=xtb)

    test_mol.solvent = 0
    with pytest.raises(ex.SolventUnavailable):
        _ = get_solvent_name(test_mol, method=xtb)

    # return to the gas phase
    test_mol.solvent = None


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
