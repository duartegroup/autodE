import os
from autode import Molecule, Atom, Calculation
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


def test_thermochemistry_h2o():

    h2o = Molecule(smiles='O')
    calc = Calculation(name='tmp',
                       molecule=h2o,
                       method=orca,
                       keywords=orca.keywords.hess)
    calc.output.filename = 'H2O_hess_orca.out'

    h2o.calc_thermo(calc=calc)

    print(h2o.energies)
