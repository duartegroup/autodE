import os
import numpy as np
from autode import Molecule, Atom, Calculation
from autode.values import Energy
from autode.species import Species
from autode.methods import ORCA
from . import testutils
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
def test_thermochemistry_h2o():

    h2o = Molecule(smiles='O')

    calc = Calculation(name='tmp',
                       molecule=h2o,
                       method=orca,
                       keywords=orca.keywords.hess)
    calc.output.filename = 'H2O_hess_orca.out'

    # Calculate using the default method from ORCA
    h2o.calc_thermo(calc=calc, ss='1atm', sn=1)

    # Ensure the calculated free energy contribution is close to value obtained
    # directly from ORCA
    assert np.isclose(h2o.g_cont, Energy(0.00327564, units='ha'),
                      atol=Energy(0.1, units='kcal mol-1').to('ha'))

    # and likewise for the enthalpy
    assert np.isclose(h2o.h_cont, Energy(0.02536189087, units='ha'),
                      atol=Energy(0.1, units='kcal mol-1').to('ha'))


def test_thermochem_single_atom():

    f_entropy_g09 = Energy(0.011799 / 298.15, units='Ha')  # T S from g09

    f_atom = Molecule(atoms=[Atom('F')])
    f_atom.calc_thermo()
    f_entropy = (f_atom.h_cont - f_atom.g_cont) / 298.15

    # Ensure the calculated and 'actual' from Gaussian09 are close
    assert np.isclose(f_entropy_g09, f_entropy, atol=2E-5)
