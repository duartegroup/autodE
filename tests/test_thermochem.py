import os
from autode import Molecule
from autode.thermo.symmetry import symmetry_number
from . import testutils
here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'symm.zip'))
def test_symmetry_number():

    assert symmetry_number(species=Molecule('BH3.xyz')) == 6
    assert symmetry_number(species=Molecule('C6H6.xyz')) == 12
    assert symmetry_number(species=Molecule('CO.xyz')) == 1
    assert symmetry_number(species=Molecule('CO2.xyz')) == 2
    assert symmetry_number(species=Molecule('H2O.xyz')) == 2
    assert symmetry_number(species=Molecule('H3N.xyz')) == 3
