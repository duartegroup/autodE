import os
import pytest
from .. import testutils
from autode.atoms import Atom
from autode.species import Molecule
from autode.wrappers.ORCA import ORCA
from autode.wrappers.keywords import OptKeywords
from autode.pes.relaxed import RelaxedPESnD
from autode.pes.pes_nd import _energy_unit_from_name
from autode.units import Unit
here = os.path.dirname(os.path.abspath(__file__))


def test_points_gen_idxs_1d():

    pes1d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})

    expected_points = [[(0,)], [(1,)], [(2,)]]
    for expected, true in zip(expected_points, pes1d._points_generator()):
        assert expected == true


def test_points_gen_idxs_2d():
    pes2d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2),
                             (1, 2): (1.0, 2.0, 2)})

    assert len(list(pes2d._points())) == 4

    # For a 2D grid there is points with indices that sum to 0, 1 and 2
    expected_points = [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]]

    for expected, true in zip(expected_points, pes2d._points_generator()):
        # order doesn't matter, so convert to sets
        assert set(expected) == set(true)


def test_points_gen_idxs_3d():
    pes3d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2),
                             (1, 2): (1.0, 2.0, 2),
                             (2, 3): (1.0, 2.0, 2)})

    assert pes3d.shape == (2, 2, 2)

    expected_points = [[(0, 0, 0)],
                       [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
                       [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
                       [(1, 1, 1)]]

    for expected, true in zip(expected_points, pes3d._points_generator()):
        # order doesn't matter, so convert to sets
        assert set(expected) == set(true)


@testutils.work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_relaxed_with_keywords():

    pes = RelaxedPESnD(Molecule(atoms=[Atom('H'), Atom('H', x=0.70)]),
                       rs={(0, 1): (1.5, 5)})

    orca = ORCA()

    # Spoof ORCA availability
    orca.path = here
    assert orca.available

    pes.calculate(method=orca,
                  keywords=OptKeywords(['PBE', 'def2-SVP', 'LooseOpt']))

    # Ensure the PES has been populated, using the saved output files
    assert all(e < -1 for e in pes._energies)

    # Ensure the correct keywords have been used *NOTE* needs whitespace
    assert os.path.exists('H2_scan_0_orca.inp')
    assert 'pbe ' in open('H2_scan_0_orca.inp', 'r').readline().lower()


def test_calculate_single_without_est():

    pes = RelaxedPESnD(Molecule(atoms=[Atom('H'), Atom('H', x=0.70)]),
                       rs={(0, 1): (1.5, 5)})

    orca = ORCA()
    orca.path = '/a/path/that/does/not/exist'
    assert not orca.available

    # Cannot calculate a surface without a working method
    with pytest.raises(RuntimeError):
        pes.calculate(method=orca)


def test_units_name_to_units():

    unit = _energy_unit_from_name('eV')
    assert isinstance(unit, Unit)
    assert unit.name.lower() == 'ev'

    with pytest.raises(Exception):
        _ = _energy_unit_from_name('ang')
