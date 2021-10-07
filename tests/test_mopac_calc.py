from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.MOPAC import get_keywords
from autode.exceptions import CouldNotGetProperty, UnsuppportedCalculationInput
from autode.calculation import Calculation, CalculationInput
from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.constants import Constants
from autode.config import Config
from autode.point_charges import PointCharge
from . import testutils
import numpy as np
import os
import pytest

here = os.path.dirname(os.path.abspath(__file__))
method = MOPAC()

methylchloride = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]',
                          solvent_name='water')


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mopac.zip'))
def test_mopac_opt_calculation():

    calc = Calculation(name='opt', molecule=methylchloride,
                       method=method, keywords=Config.MOPAC.keywords.opt)
    calc.run()

    assert os.path.exists('opt_mopac.mop') is True
    assert os.path.exists('opt_mopac.out') is True
    assert len(calc.get_final_atoms()) == 5

    # Actual energy in Hartrees
    energy = Constants.eV_to_ha * -430.43191
    assert energy - 0.0001 < calc.get_energy() < energy + 0.0001

    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.input.filename == 'opt_mopac.mop'
    assert calc.output.filename == 'opt_mopac.out'
    assert calc.terminated_normally
    assert calc.optimisation_converged() is True

    with pytest.raises(CouldNotGetProperty):
        _ = calc.get_gradients()

    with pytest.raises(NotImplementedError):
        _ = calc.optimisation_nearly_converged()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mopac.zip'))
def test_mopac_with_pc():

    calc = Calculation(name='opt_pc', molecule=methylchloride,
                       method=method,
                       keywords=Config.MOPAC.keywords.opt,
                       point_charges=[PointCharge(1, x=4, y=4, z=4)])
    calc.run()

    assert os.path.exists('opt_pc_mopac.mop') is True
    assert os.path.exists('opt_pc_mopac.out') is True
    assert len(calc.get_final_atoms()) == 5

    # Actual energy in Hartrees without any point charges
    energy = Constants.eV_to_ha * -430.43191
    assert np.abs(calc.get_energy() - energy) > 0.0001


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mopac.zip'))
def test_other_spin_states():

    o_singlet = Molecule(atoms=[Atom('O')], mult=1)
    o_singlet.name = 'molecule'

    calc = Calculation(name='O_singlet',
                       molecule=o_singlet,
                       method=method,
                       keywords=Config.MOPAC.keywords.sp)
    calc.run()
    singlet_energy = calc.get_energy()

    o_triplet = Molecule(atoms=[Atom('O')], mult=3)
    o_triplet.name = 'molecule'

    calc = Calculation(name='O_triplet',
                       molecule=o_triplet,
                       method=method,
                       keywords=Config.MOPAC.keywords.sp)
    calc.run()
    triplet_energy = calc.get_energy()

    assert triplet_energy < singlet_energy

    h_doublet = Molecule(atoms=[Atom('H')], mult=2)
    h_doublet.name = 'molecule'

    calc = Calculation(name='h',
                       molecule=h_doublet,
                       method=method,
                       keywords=Config.MOPAC.keywords.sp)
    calc.run()

    # Open shell doublet should work
    assert calc.get_energy() is not None

    h_quin = Molecule(atoms=[Atom('H')], mult=5)
    h_quin.name = 'molecule'

    with pytest.raises(UnsuppportedCalculationInput):
        calc = Calculation(name='h',
                           molecule=h_quin,
                           method=method,
                           keywords=Config.MOPAC.keywords.sp)
        calc.run()

    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mopac.zip'))
def test_bad_geometry():

    # Calculation with the wrong spin state should fail
    calc = Calculation(name='h2_overlap_opt',
                       molecule=Molecule(atoms=[Atom('H'), Atom('H')]),
                       method=method,
                       keywords=Config.MOPAC.keywords.opt)

    calc.output.filename = 'h2_overlap_opt_mopac.out'

    assert not calc.terminated_normally
    assert calc.get_energy() is None
    assert not calc.optimisation_converged()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mopac.zip'))
def test_constrained_opt():

    methane = Molecule(name='methane', smiles='C')

    calc = Calculation(name='methane_opt', molecule=methane,
                       method=method,
                       keywords=Config.MOPAC.keywords.opt)
    calc.run()
    opt_energy = calc.get_energy()

    # Constrained optimisation with a C–H distance of 1.2 Å
    # (carbon is the first atom in the file)
    const = Calculation(name='methane_const', molecule=methane,
                        method=method,
                        keywords=Config.MOPAC.keywords.opt,
                        distance_constraints={(0, 1): 1.2})
    const.run()

    assert opt_energy < const.get_energy()

    with pytest.raises(Exception):
        _ = calc.get_hessian()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mopac.zip'))
def test_grad():

    h2 = Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=0.5)])

    grad_calc = Calculation(name='h2_grad',
                            molecule=h2,
                            method=method,
                            keywords=Config.MOPAC.keywords.grad)
    grad_calc.run()
    energy = grad_calc.get_energy()
    assert energy is not None

    gradients = grad_calc.get_gradients()
    assert gradients.shape == (2, 3)

    delta_r = 1E-5
    h2_disp = Molecule(name='h2_disp',
                       atoms=[Atom('H'), Atom('H', x=0.5 + delta_r)])
    h2_disp.single_point(method)

    delta_energy = h2_disp.energy - energy      # Ha]
    grad = delta_energy / delta_r               # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(gradients[1, 0] - grad) < 1E-1

    # Broken gradient file
    grad_calc_broken = Calculation(name='h2_grad',
                                   molecule=h2,
                                   method=method,
                                   keywords=Config.MOPAC.keywords.grad)
    grad_calc_broken.output.filename = 'h2_grad_broken.out'

    with pytest.raises(CouldNotGetProperty):
        _ = grad_calc_broken.get_gradients()


def test_termination_short():

    calc = Calculation(name='test', molecule=methylchloride,
                       method=method, keywords=Config.MOPAC.keywords.sp)

    calc.output.filename = 'test.out'
    with open(calc.output.filename, 'w') as test_output:
        print('JOB ENDED NORMALLY', 'another line', sep='\n', file=test_output)

    assert calc.terminated_normally
    os.remove(calc.output.filename)


def test_mopac_keywords():

    calc_input = CalculationInput(keywords=Config.MOPAC.keywords.sp,
                                  added_internals=None,
                                  additional_input=None,
                                  point_charges=None)

    keywords = get_keywords(calc_input=calc_input, molecule=methylchloride)
    assert any('1scf' == kw.lower() for kw in keywords)

    calc_input.keywords = Config.MOPAC.keywords.grad
    keywords = get_keywords(calc_input=calc_input, molecule=methylchloride)
    assert any('grad' == kw.lower() for kw in keywords)

    h = Molecule(name='H', smiles='[H]')
    assert h.mult == 2

    keywords = get_keywords(calc_input=calc_input, molecule=h)
    assert any('doublet' == kw.lower() for kw in keywords)


def test_get_version_no_output():

    calc = Calculation(name='test',
                       molecule=methylchloride,
                       method=method,
                       keywords=method.keywords.sp)
    calc.output.filename = 'test.out'

    with open(calc.output.filename, 'w') as test_output:
        print('some', 'incorrect', 'lines', sep='\n', file=test_output)

    assert not calc.terminated_normally

    version = method.get_version(calc)
    assert version == '???'

    os.remove(calc.output.filename)
