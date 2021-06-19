import os
import pytest
import numpy as np
from autode import Molecule, Atom, Calculation
from autode.thermochemistry import calculate_thermo_cont
from autode.values import Energy
from autode.species import Species
from autode.methods import ORCA
from . import testutils
from autode.thermochemistry.igm import (_q_rot_igm, _s_rot_rr)

here = os.path.dirname(os.path.abspath(__file__))

orca = ORCA()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'symm.zip'))
def test_symmetry_number():

    assert Molecule().symmetry_number == 1

    assert Molecule('BH3.xyz').symmetry_number == 6
    assert Molecule('C6H6.xyz').symmetry_number == 12
    assert Molecule('CO.xyz').symmetry_number == 1
    assert Molecule('CO2.xyz').symmetry_number == 2
    assert Molecule('H2O.xyz').symmetry_number == 2
    assert Molecule('H3N.xyz').symmetry_number == 3

    # Symmetry numbers aren't calculated for large molecules
    h_100 = Species('tmp', atoms=100*[Atom('H')], charge=1, mult=1)
    assert h_100.symmetry_number == 1


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'thermochem.zip'))
def test_h2o():

    h2o = Molecule(smiles='O')

    calc = Calculation(name='tmp',
                       molecule=h2o,
                       method=orca,
                       keywords=orca.keywords.hess)
    calc.output.filename = 'H2O_hess_orca.out'
    assert calc.output.exists

    # Calculate using the default method from ORCA
    h2o.calc_thermo(calc=calc, ss='1atm', sn=1)

    # Ensure the calculated free energy contribution is close to value obtained
    # directly from ORCA
    assert h2o.g_cont is not None
    assert np.isclose(h2o.g_cont, Energy(0.00327564, units='ha'),
                      atol=Energy(0.1, units='kcal mol-1').to('ha'))

    # and likewise for the enthalpy
    assert np.isclose(h2o.h_cont, Energy(0.02536189087, units='ha'),
                      atol=Energy(0.1, units='kcal mol-1').to('ha'))

    # Check that the standard state correction is applied correctly
    h2o_1m = Molecule(smiles='O')
    h2o_1m.calc_thermo(calc=calc, ss='1M', sn=1)

    # with a difference of ~1.9 kcal mol-1 at room temperature
    g_diff = (h2o_1m.g_cont - h2o.g_cont).to('kcal mol-1')
    assert np.isclose(g_diff - Energy(1.9, units='kcal mol-1'), 0.0,
                      atol=0.2)

    # but the enthalpy is the same
    assert np.isclose(h2o_1m.h_cont, h2o.h_cont, atol=1E-6)

    # Cannot calculate any other standard states
    with pytest.raises(ValueError):
        h2o.calc_thermo(calc=calc, ss='1nm3', sn=1)


def test_single_atom():

    f_entropy_g09 = Energy(0.011799 / 298.15, units='Ha')  # T S from g09

    f_atom = Molecule(atoms=[Atom('F')])
    f_atom.calc_thermo()
    f_entropy = (f_atom.h_cont - f_atom.g_cont) / 298.15

    # Ensure the calculated and 'actual' from Gaussian09 are close
    assert np.isclose(f_entropy_g09, f_entropy, atol=2E-5)

    # Ensure the rotational partition functions and entropy are 1 and 0
    assert np.isclose(_q_rot_igm(f_atom, temp=298, sigma_r=0), 1.0)
    assert np.isclose(_s_rot_rr(f_atom, temp=298, sigma_r=0), 0.0)


def test_no_atoms():

    mol = Molecule()
    assert mol.g_cont is None and mol.h_cont is None

    # Nothing to be calculated for a molecule with no atoms
    calculate_thermo_cont(mol)
    assert mol.g_cont is None and mol.h_cont is None


def test_no_frequencies():

    mol = Molecule(smiles='O')

    # Cannot calculate the virbational component without vibrational
    # frequencies
    with pytest.raises(ValueError):
        calculate_thermo_cont(mol)


def test_linear_non_linear_rot():

    h2_tri = Molecule(atoms=[Atom('H'), Atom('H', x=1), Atom('H', x=1, y=1)])
    h2_lin = Molecule(atoms=[Atom('H'), Atom('H', x=1), Atom('H', x=2)])

    assert h2_lin.is_linear()

    # Non linear molecules have slightly more entropy than linear ones
    assert _s_rot_rr(h2_tri, temp=298, sigma_r=1) > _s_rot_rr(h2_lin, temp=298, sigma_r=1)
