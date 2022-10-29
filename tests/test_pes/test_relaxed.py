import os
import numpy as np
import pytest
from .. import testutils
from autode.atoms import Atom
from autode.species import Molecule
from autode.wrappers.ORCA import ORCA
from autode.wrappers.keywords import OptKeywords
from autode.pes.relaxed import RelaxedPESnD as PESnD
from autode.units import Unit, energy_unit_from_name
from autode.utils import work_in_tmp_dir

here = os.path.dirname(os.path.abspath(__file__))


class RelaxedPESnD(PESnD):
    def __init__(self, species=None, rs=None):
        super(RelaxedPESnD, self).__init__(species=species, rs=rs)


def test_points_gen_idxs_1d():

    pes1d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 3)})

    expected_points = [[(0,)], [(1,)], [(2,)]]
    for expected, true in zip(expected_points, pes1d._points_generator()):
        assert expected == true


def test_points_gen_idxs_2d():
    pes2d = RelaxedPESnD(rs={(0, 1): (1.0, 2.0, 2), (1, 2): (1.0, 2.0, 2)})

    assert len(list(pes2d._points())) == 4

    # For a 2D grid there is points with indices that sum to 0, 1 and 2
    expected_points = [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]]

    for expected, true in zip(expected_points, pes2d._points_generator()):
        # order doesn't matter, so convert to sets
        assert set(expected) == set(true)


def test_points_gen_idxs_3d():
    pes3d = RelaxedPESnD(
        rs={
            (0, 1): (1.0, 2.0, 2),
            (1, 2): (1.0, 2.0, 2),
            (2, 3): (1.0, 2.0, 2),
        }
    )

    assert pes3d.shape == (2, 2, 2)

    expected_points = [
        [(0, 0, 0)],
        [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
        [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
        [(1, 1, 1)],
    ]

    for expected, true in zip(expected_points, pes3d._points_generator()):
        # order doesn't matter, so convert to sets
        assert set(expected) == set(true)


@testutils.work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_relaxed_with_keywords():

    pes = RelaxedPESnD(
        Molecule(atoms=[Atom("H"), Atom("H", x=0.70)]), rs={(0, 1): (1.5, 5)}
    )

    orca = ORCA()

    # Spoof ORCA availability
    orca.path = here
    assert orca.is_available

    pes.calculate(
        method=orca, keywords=OptKeywords(["PBE", "def2-SVP", "LooseOpt"])
    )

    # Ensure the PES has been populated, using the saved output files
    assert all(e < -1 for e in pes._energies)

    # Ensure the correct keywords have been used *NOTE* needs whitespace
    assert os.path.exists("H2_scan_0_orca.inp")
    assert "pbe " in open("H2_scan_0_orca.inp", "r").readline().lower()


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calculate_single_without_est():

    pes = RelaxedPESnD(
        Molecule(atoms=[Atom("H"), Atom("H", x=0.70)]), rs={(0, 1): (1.5, 5)}
    )

    orca = ORCA()
    orca.path = "/a/path/that/does/not/exist"
    assert not orca.is_available

    # Cannot calculate a surface without a working method
    with pytest.raises(RuntimeError):
        pes.calculate(method=orca)


def test_units_name_to_units():

    unit = energy_unit_from_name("eV")
    assert isinstance(unit, Unit)
    assert unit.name.lower() == "ev"

    with pytest.raises(Exception):
        _ = energy_unit_from_name("ang")


@testutils.work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_sn2_ts_guesses():

    reac = Molecule(
        name="reac",
        charge=-1,
        mult=1,
        atoms=[
            Atom("F", -6.16710, 4.34010, 0.14884),
            Atom("Cl", -0.96539, 4.54693, -0.06342),
            Atom("C", -2.73988, 4.47792, 0.00591),
            Atom("H", -3.09044, 3.88869, -0.83567),
            Atom("H", -3.02574, 4.01317, 0.94432),
            Atom("H", -3.12217, 5.49228, -0.05113),
        ],
    )

    # Construct the 2D PES from the current C-F and C-Cl distances to the
    # ones at the product, hopefully over the TS
    pes = RelaxedPESnD(
        species=reac, rs={(0, 2): (1.45, 10), (2, 1): (2.5, 10)}
    )

    pes.load("sn2_pes.npz")

    ts_guesses = list(pes.ts_guesses())
    assert len(ts_guesses) == 1

    ts_guess = ts_guesses[0]
    assert np.isclose(ts_guess.distance(0, 2).to("Å"), 1.891, atol=1e-3)  # C-F

    assert np.isclose(
        ts_guess.distance(2, 1).to("Å"), 2.179, atol=1e-3
    )  # C-Cl


@testutils.work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_da_ts_guesses():

    cyclohexene = Molecule(
        atoms=[
            Atom("C", -1.25524, 0.55843, -0.45127),
            Atom("C", -0.11901, 1.52914, -0.34083),
            Atom("C", 1.12810, 1.05947, -0.22343),
            Atom("C", 1.36167, -0.42098, -0.26242),
            Atom("C", 0.31173, -1.21999, 0.53472),
            Atom("C", -1.08411, -0.56797, 0.57151),
            Atom("H", -1.28068, 0.12348, -1.46969),
            Atom("H", -2.22404, 1.06103, -0.30930),
            Atom("H", -0.31910, 2.60539, -0.34475),
            Atom("H", 1.98105, 1.73907, -0.13515),
            Atom("H", 2.37438, -0.67767, 0.08526),
            Atom("H", 1.32299, -0.74206, -1.32088),
            Atom("H", 0.25137, -2.23441, 0.11110),
            Atom("H", 0.67503, -1.34778, 1.56617),
            Atom("H", -1.86306, -1.33147, 0.42039),
            Atom("H", -1.26117, -0.13357, 1.56876),
        ]
    )

    pes = RelaxedPESnD(
        species=cyclohexene,
        rs={(0, 5): (1.45, 3.0, 10), (3, 4): (1.45, 3.0, 10)},
    )

    pes.load("da_pes.npz")
    ts_guesses = list(pes.ts_guesses())
    assert len(ts_guesses) > 0

    def has_correct_dists(mol):
        return np.isclose(mol.distance(0, 5), 2.311, atol=1e-3) and np.isclose(
            mol.distance(3, 4), 2.311, atol=1e-3
        )

    # Diels-Alder TS should be symmetric, and for this surface the bond lengths
    # ~2.3 Å
    assert any(has_correct_dists(ts_guess) for ts_guess in ts_guesses)


@testutils.work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_1d_pes_acetone_cn():

    species = Molecule("acetone_cn.xyz", charge=-1, solvent_name="water")
    pes = RelaxedPESnD(species=species, rs={(1, 10): (1.5, 15)})

    pes.load("acetone_cn.npz")

    # Should only have a single TS guess on the surface
    ts_guesses = list(pes.ts_guesses())
    assert len(ts_guesses) == 1

    ts_guess = ts_guesses[0]  # Check the distance is close to the true value
    assert np.isclose(ts_guess.distance(1, 10).to("Å"), 1.919, atol=0.1)
