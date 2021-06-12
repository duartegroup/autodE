import os
from autode import Molecule, Atom
from . import testutils
here = os.path.dirname(os.path.abspath(__file__))


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
    assert Molecule(atoms=100*[Atom('H')]).symmetry_number == 1
