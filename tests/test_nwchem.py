from autode.wrappers.NWChem import NWChem, ecp_block, get_keywords
from autode.calculation import Calculation, CalculationInput
from autode.exceptions import CouldNotGetProperty, CalculationException
from autode.species.molecule import Molecule
from autode.wrappers.keywords import OptKeywords, MaxOptCycles, SinglePointKeywords
from autode.wrappers.basis_sets import def2svp
from autode.wrappers.wf import hf
from autode.wrappers.functionals import pbe0
from autode.atoms import Atom
from . import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = NWChem()

opt_keywords = OptKeywords(['driver\n gmax 0.002\n  grms 0.0005\n'
                            '  xmax 0.01\n   xrms 0.007\n  eprec 0.00003\nend',
                            'basis\n  *   library Def2-SVP\nend',
                            'dft\n   xc xpbe96 cpbe96\nend',
                            'task dft optimize'])


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'nwchem.zip'))
def test_opt_calc():

    calc = Calculation(name='opt', molecule=test_mol, method=method,
                       keywords=opt_keywords)
    calc.run()

    assert os.path.exists('opt_nwchem.nw')
    assert os.path.exists('opt_nwchem.out')

    final_atoms = calc.get_final_atoms()
    assert len(final_atoms) == 5
    assert type(final_atoms[0]) is Atom
    assert -40.4165 < calc.get_energy() < -40.4164
    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.input.filename == 'opt_nwchem.nw'
    assert calc.output.filename == 'opt_nwchem.out'
    assert calc.terminated_normally
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False

    # No Hessian is computed for an optimisation calculation
    with pytest.raises(CouldNotGetProperty):
        _ = calc.get_hessian()

    charges = calc.get_atomic_charges()
    assert len(charges) == 5
    assert all(-1.0 < c < 1.0 for c in charges)

    # Optimisation should result in small gradients
    gradients = calc.get_gradients()
    assert len(gradients) == 5
    assert all(-0.1 < np.linalg.norm(g) < 0.1 for g in gradients)


def test_opt_single_atom():

    h = Molecule(name='H', smiles='[H]')
    calc = Calculation(name='opt_h', molecule=h, method=method,
                       keywords=opt_keywords)
    calc.generate_input()

    # Can't do an optimisation of a hydrogen atom..
    assert os.path.exists('opt_h_nwchem.nw')
    input_lines = open('opt_h_nwchem.nw', 'r').readlines()
    assert 'opt' not in [keyword.lower() for keyword in input_lines[0].split()]

    os.remove('opt_h_nwchem.nw')


def test_exception_wf_solvent_calculation():

    solvated_mol = Molecule(name='methane', smiles='C',
                            solvent_name='water')

    calc = Calculation(name='opt',
                       molecule=solvated_mol,
                       method=method,
                       keywords=SinglePointKeywords([hf, def2svp]))

    # Cannot have solvent with a non-DFT calculation(?)
    with pytest.raises(CalculationException):
        calc.generate_input()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'nwchem.zip'))
def test_ecp_calc():

    # Should have no ECP block for molecule with only light elements
    water_ecp_block = ecp_block(Molecule(smiles='O'),
                                keywords=method.keywords.sp)
    assert water_ecp_block == ''

    # Should have no ECP block if the keywords do not define an ECP
    pd_ecp_block = ecp_block(Molecule(smiles='[Pd]'), keywords=OptKeywords([]))
    assert pd_ecp_block == ''

    pdh2 = Molecule(smiles='[H][Pd][H]', name='H2Pd')
    pdh2.single_point(method=method)

    assert os.path.exists('H2Pd_sp_nwchem.nw')
    input_lines = open('H2Pd_sp_nwchem.nw', 'r').readlines()
    assert any('ecp' in line for line in input_lines)

    assert pdh2.energy is not None


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'nwchem.zip'))
def test_opt_hf_constraints():

    keywords = OptKeywords(['driver\n gmax 0.002\n  grms 0.0005\n'
                            '  xmax 0.01\n   xrms 0.007\n  eprec 0.00003\nend',
                            'basis\n  *   library Def2-SVP\nend',
                            'task scf optimize'])

    h2o = Molecule(name='water', smiles='O')
    calc = Calculation(name='opt_water', molecule=h2o, method=method,
                       keywords=keywords,
                       cartesian_constraints=[0],
                       distance_constraints={(0, 1): 0.95})
    calc.run()
    h2o.atoms = calc.get_final_atoms()
    assert 0.94 < h2o.distance(0, 1) < 0.96


def test_get_keywords_max_opt_cyles():

    opt_block = ('driver\n'
                 '  gmax 0.0003\n'
                 '  maxiter 100\n'
                 'end')

    # Defining the maximum number of optimisation cycles should override the
    # value set in the driver
    kwds = OptKeywords([opt_block, def2svp, pbe0, MaxOptCycles(10),
                        'task dft optimize',
                        'task dft property'])

    calc_input = CalculationInput(keywords=kwds)

    # Should only have a single instance of the maxiter declaration
    str_keywords = get_keywords(calc_input, molecule=test_mol)
    modified_opt_block = str_keywords[0].split('\n')

    assert sum('maxiter' in line for line in modified_opt_block) == 1

    # and should be 10 not 100
    assert sum('maxiter 100' in line for line in modified_opt_block) == 0
    assert sum('maxiter 10' in line for line in modified_opt_block) == 1

    # Also if the maxiter is not defined already

    kwds = OptKeywords([('driver\n  gmax 0.0003\nend'), def2svp, pbe0, MaxOptCycles(10)])
    calc_input = CalculationInput(keywords=kwds)
    modified_opt_block2 = get_keywords(calc_input, molecule=test_mol)[0].split('\n')

    assert sum('maxiter 10' in line for line in modified_opt_block2) == 1


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'nwchem.zip'))
def test_hessian_extract_ts():

    atoms = [Atom('F',   0.00000, 0.00000,  2.50357),
             Atom('Cl', -0.00000, 0.00000, -1.62454),
             Atom('C',   0.00000, 0.00000,  0.50698),
             Atom('H',   1.05017, 0.24818,  0.60979),
             Atom('H', -0.74001,  0.78538,  0.60979),
             Atom('H', -0.31016, -1.03356,  0.60979)]

    calc = Calculation(name='sn2_hess',
                       molecule=Molecule(name='ts', atoms=atoms),
                       keywords=method.keywords.hess,
                       method=method)
    calc.output.filename = 'sn2_hess_nwchem.out'

    hess = calc.get_hessian()
    assert hess.shape == (3*len(atoms), 3*len(atoms))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'nwchem.zip'))
def test_hessian_extract_butane():

    calc = Calculation(name='butane',
                       molecule=Molecule('butane.xyz'),
                       keywords=method.keywords.hess,
                       method=method)
    calc.output.filename = 'butane_hess_nwchem.out'

    hess = calc.get_hessian()

    # bottom right corner element should be positive
    assert hess[-1, -1] > 0
    assert np.isclose(hess.frequencies[0].to('cm-1'),
                      -2385.13,
                      atol=3.0)

    assert np.isclose(hess.frequencies[-1].to('cm-1'),
                      3500.27,
                      atol=3.0)
