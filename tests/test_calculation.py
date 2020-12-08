from autode.calculation import Calculation, get_solvent_name
from autode.solvent.solvents import get_solvent
from autode.methods import XTB
from autode.species import Molecule
import autode.exceptions as ex
from autode.utils import work_in_tmp_dir
import pytest
import os

test_mol = Molecule(smiles='O', name='test_mol')
xtb = XTB()


def test_calc_class():

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

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_gradients()

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

    calc = Calculation(name='tmp',
                       molecule=test_mol,
                       method=xtb,
                       keywords=xtb.keywords.sp)
    calc._fix_unique()
    assert calc.name == 'tmp_xtb'

    assert os.path.exists('.autode_calculations')
    assert len(open('.autode_calculations', 'r').readlines()) == 1


def test_solvent_get():

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

    # return to the gas phase
    test_mol.solvent = None



