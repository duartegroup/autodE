import os
import numpy as np
from .. import testutils
from .sample_pes import TestPES
from autode.pes.mep import peak_point
from autode.pes.relaxed import RelaxedPESnD
from autode import Molecule, Atom

here = os.path.dirname(os.path.abspath(__file__))


def flat_h2_pes():
    """Flat H2 PES with shape (11,)"""

    pes = TestPES(
        rs={(0, 1): np.linspace(-1, 1, num=11)},
        species=Molecule(atoms=[Atom("H"), Atom("H", x=0.7)]),
    )
    pes._energies.fill(-1.0)
    pes._coordinates = np.zeros(shape=(11, 2, 3))
    pes._coordinates[:, 1, 0] = 0.7  # Set all x values for H_b to 0.7 Å

    return pes


def test_simple_peak():

    pes = TestPES(
        rs={
            (0, 1): np.linspace(-1, 1, num=11),
            (1, 2): np.linspace(-1, 1, num=11),
        }
    )

    p = peak_point(
        energies=0.01 * (pes.r1**2 - pes.r2**2),
        point1=(5, 0),
        point2=(5, 10),
    )

    assert p == (5, 5)


@testutils.work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_sn2_ts_guesses():

    r = Molecule(
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

    pes = RelaxedPESnD(species=r, rs={(0, 2): (1.45, 10), (2, 1): (2.5, 10)})
    pes.load("sn2_pes.npz")

    prod = Molecule(
        name="prod",
        charge=-1,
        mult=1,
        atoms=[
            Atom("F", -4.94585, 4.39058, 0.09591),
            Atom("Cl", -1.00002, 4.54623, -0.06354),
            Atom("C", -3.50010, 4.44743, 0.03750),
            Atom("H", -3.23304, 3.87507, -0.83916),
            Atom("H", -3.16714, 4.00107, 0.96305),
            Atom("H", -3.26455, 5.49873, -0.04490),
        ],
    )

    # Specifying the product should use a different algorithm, but return the
    # same TS guess geometry
    ts_guesses = list(pes.ts_guesses(product=prod))
    assert len(ts_guesses) == 1
    ts_guess = ts_guesses[0]

    assert np.isclose(ts_guess.distance(0, 2).to("Å"), 1.891, atol=0.1)
    assert np.isclose(ts_guess.distance(2, 1).to("Å"), 2.179, atol=0.1)


def test_mep_ts_guess_no_graph():

    pes = flat_h2_pes()

    h2_no_graph = Molecule(atoms=[Atom("H"), Atom("H", x=1.0)])
    h2_no_graph.graph = None

    # Cannot find TS guesses from a product if it doesn't have a graph
    assert len(list(pes.ts_guesses(product=h2_no_graph))) == 0


def test_mep_ts_guess_no_isomorphism():

    pes = flat_h2_pes()

    h2_no_bond = Molecule(atoms=[Atom("H"), Atom("H", x=2.0)])
    assert len(h2_no_bond.graph.edges) == 0

    # Cannot find TS guess if the product is not isomorphic to any point on the
    # surface
    assert len(list(pes.ts_guesses(product=h2_no_bond))) == 0
