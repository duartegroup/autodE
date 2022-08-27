import numpy as np
import os
import pytest
from autode.atoms import Atom
from autode.wrappers.XTB import XTB
from autode.calculations import Calculation
from autode.species.molecule import Molecule
from autode.point_charges import PointCharge
from autode.exceptions import AtomsNotFound, CalculationException
from autode.config import Config
from .. import testutils

here = os.path.dirname(os.path.abspath(__file__))

method = XTB()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_xtb_calculation():

    test_mol = Molecule(name='test_mol',
                        smiles='O=C(C=C1)[C@@](C2NC3C=C2)([H])[C@@]3([H])C1=O')
    calc = Calculation(name='opt', molecule=test_mol, method=method,
                       keywords=Config.XTB.keywords.opt)
    calc.run()

    assert os.path.exists('opt_xtb.xyz') is True
    assert os.path.exists('opt_xtb.out') is True
    assert test_mol.n_atoms == 22
    assert test_mol.energy == -36.990267613593
    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.input.filename == 'opt_xtb.xyz'
    assert calc.output.filename == 'opt_xtb.out'
    assert calc.optimiser.converged

    with pytest.raises(NotImplementedError):
        _ = calc.optimiser.last_energy_change

    charges = test_mol.partial_charges
    assert len(charges) == 22
    assert all(-1.0 < c < 1.0 for c in charges)

    test_mol.constraints.update(distance={(0, 1): 1.2539792},
                                cartesian=[0])

    const_opt = Calculation(name='const_opt',
                            molecule=test_mol,
                            method=method,
                            keywords=Config.XTB.keywords.opt)

    const_opt.generate_input()
    assert os.path.exists('const_opt_xtb.xyz')
    assert os.path.exists('xcontrol_const_opt_xtb')

    const_opt.clean_up(force=True)
    assert not os.path.exists('xcontrol_const_opt_xtb')

    # Write an empty output file
    open('tmp.out', 'w').close()
    const_opt.output.filename = 'tmp.out'

    # cannot get atoms from an empty file
    with pytest.raises(AtomsNotFound):
        const_opt._executor.set_properties()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_energy_extract_no_energy():

    calc = Calculation(name='opt',
                       molecule=Molecule(smiles='[H][H]'),
                       method=method,
                       keywords=Config.XTB.keywords.sp)

    # Output where the energy is not present
    calc.output.filename = 'h2_sp_xtb_no_energy.out'

    with pytest.raises(CalculationException):
        calc._executor.set_properties()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_point_charge():

    test_mol = Molecule(name='test_mol', smiles='C')

    # Methane with a point charge fairly far away
    calc = Calculation(name='opt_point_charge',
                       molecule=test_mol,
                       method=method,
                       keywords=Config.XTB.keywords.opt,
                       point_charges=[PointCharge(charge=1.0, x=10, y=1, z=1)])
    calc.run()

    assert -4.178 < test_mol.energy < -4.175


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_gradients():

    h2 = Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.0)])
    h2.single_point(method)

    delta_r = 1E-5
    h2_disp = Molecule(name='h2_disp',
                       atoms=[Atom('H'), Atom('H', x=1.0 + delta_r)])
    h2_disp.single_point(method)

    delta_energy = h2_disp.energy - h2.energy   # Ha
    grad = delta_energy / delta_r               # Ha A^-1

    calc = Calculation(name='h2_grad', molecule=h2,
                            method=method,
                            keywords=method.keywords.grad)

    calc.run()

    diff = h2.gradient[1, 0] - grad    # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1E-5

    # Older xtb version
    with open(f'methane_OLD.grad', 'w') as gradient_file:
        print('$gradient\n'
              'cycle =      1    SCF energy =    -4.17404780397   |dE/dxyz| =  0.027866\n'
              '3.63797523123375     -1.13138130908142     -0.00032759661848      C \n'
              '5.72449332438353     -1.13197561185651      0.00028950521969      H \n'
              ' 2.94133258016711      0.22776472016180     -1.42078243039077      H \n'
              ' 2.94175598539510     -0.58111835182372      1.88747566982948      H \n'
              '2.94180792167968     -3.04156357656436     -0.46665514803992      H \n'
              '-1.7221823521705E-05   7.9930724499610E-05  -1.1737079840097E-04\n'
              ' 1.4116296505865E-02  -4.0359524399270E-05   3.9719638516747E-05\n'
              '-4.7199424681741E-03   9.0086220034949E-03  -9.4114548523723E-03\n'
              '-4.6956970257351E-03   3.6356853660431E-03   1.2558467871909E-02\n'
              ' -4.6834351884340E-03  -1.2683878569638E-02  -3.0693618596526E-03\n'
              '$end', file=gradient_file)

    calc = Calculation(name='methane',
                       molecule=Molecule(name='methane', smiles='C'),
                       method=method,
                       keywords=method.keywords.grad)
    gradients = method.gradient_from(calc)

    assert gradients.shape == (5, 3)
    assert np.abs(gradients[0, 0]) < 1E-3


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_xtb_6_3_2():

    mol = Molecule(name='CH3Cl', smiles='ClC')
    calc = Calculation(name='test',
                       molecule=mol,
                       method=method,
                       keywords=method.keywords.opt)

    calc.set_output_filename('xtb_6_3_2_opt.out')
    assert mol.n_atoms == 5 
    assert np.isclose(mol.atoms[-2].coord[1], -0.47139030225766)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_xtb_6_1_old():

    mol = Molecule(name='methane', smiles='C')
    calc = Calculation(name='test',
                       molecule=mol,
                       method=method,
                       keywords=method.keywords.opt)

    # TODO: check this extracts the right numbers
    for filename in ('xtb_6_1_opt.out', 'xtb_no_version_opt.out'):

        calc.set_output_filename(filename)

        assert set([atom.label for atom in mol.atoms]) == {'C', 'H'}
        assert 0.9 < mol.distance(0, 1) < 1.2