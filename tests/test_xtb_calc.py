import pytest
from autode.atoms import Atom
from autode.wrappers.XTB import XTB
from autode.calculation import Calculation
from autode.species.molecule import Molecule
from autode.point_charges import PointCharge
from autode.config import Config
from . import testutils
import numpy as np
import os
here = os.path.dirname(os.path.abspath(__file__))

method = XTB()
method.available = True


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_xtb_calculation():

    os.chdir(os.path.join(here, 'data', 'xtb'))
    XTB.available = True

    test_mol = Molecule(name='test_mol',
                        smiles='O=C(C=C1)[C@@](C2NC3C=C2)([H])[C@@]3([H])C1=O')
    calc = Calculation(name='opt', molecule=test_mol, method=method,
                       keywords=Config.XTB.keywords.opt)
    calc.run()

    assert os.path.exists('opt_xtb.xyz') is True
    assert os.path.exists('opt_xtb.out') is True
    assert len(calc.get_final_atoms()) == 22
    assert calc.get_energy() == -36.990267613593
    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.input.filename == 'opt_xtb.xyz'
    assert calc.output.filename == 'opt_xtb.out'

    with pytest.raises(NotImplementedError):
        calc.optimisation_nearly_converged()
    with pytest.raises(NotImplementedError):
        calc.get_imaginary_freqs()
    with pytest.raises(NotImplementedError):
        calc.get_normal_mode_displacements(4)

    charges = calc.get_atomic_charges()
    assert len(charges) == 22
    assert all(-1.0 < c < 1.0 for c in charges)

    const_opt = Calculation(name='const_opt', molecule=test_mol,
                            method=method,
                            distance_constraints={(0, 1): 1.2539792},
                            cartesian_constraints=[0],
                            keywords=Config.XTB.keywords.opt)

    const_opt.generate_input()
    assert os.path.exists('xcontrol_const_opt_xtb')

    os.remove('const_opt_xtb.xyz')
    os.remove('xcontrol_const_opt_xtb')
    os.remove('opt_xtb.xyz')
    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_point_charge():
    os.chdir(os.path.join(here, 'data', 'xtb'))
    XTB.available = True

    test_mol = Molecule(name='test_mol', smiles='C')

    # Methane with a point charge fairly far away
    calc = Calculation(name='opt_point_charge',
                       molecule=test_mol,
                       method=method,
                       keywords=Config.XTB.keywords.opt,
                       point_charges=[PointCharge(charge=1.0, x=10, y=1, z=1)])
    calc.run()

    assert -4.178 < calc.get_energy() < -4.175
    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_gradients():
    os.chdir(os.path.join(here, 'data', 'xtb'))

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

    diff = calc.get_gradients()[1, 0] - grad    # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1E-5

    # Older xtb version
    with open('gradient', 'w') as gradient_file:
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
    gradients = method.get_gradients(calc)

    assert gradients.shape == (5, 3)
    assert np.abs(gradients[0, 0]) < 1E-3

    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'xtb.zip'))
def test_xtb_6_3_2():

    mol = Molecule(name='CH3Cl', smiles='ClC')
    calc = Calculation(name='test',
                       molecule=mol,
                       method=method,
                       keywords=method.keywords.opt)

    calc.output.filename = 'xtb_6_3_2_opt.out'
    calc.output.file_lines = open('xtb_6_3_2_opt.out', 'r').readlines()

    assert len(calc.get_final_atoms()) == 5
