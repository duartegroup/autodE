import os
import numpy as np
import pytest
from autode.atoms import Atom
from autode.methods import ORCA
from autode.species.molecule import Molecule
from autode.pes.pes_nd import PESnD
from .sample_pes import TestPES, harmonic_2d_pes

here = os.path.dirname(os.path.abspath(__file__))


def h2():
    return Molecule(atoms=[Atom("H"), Atom("H", x=0.70)])


def test_calculate():

    pes = TestPES(species=h2(), rs={(0, 1): (1.5, 10)})

    orca = ORCA()

    # Cannot calculate the energy with a _calculate method that does nothing
    with pytest.raises(RuntimeError):
        pes.calculate(method=orca)

    # and the number of TS guesses should be 0
    assert len(list(pes.ts_guesses())) == 0


def test_plot_and_save_3d():

    pes = TestPES(
        rs={
            (0, 1): np.linspace(1, 2, num=10),
            (1, 2): np.linspace(1, 2, num=10),
            (2, 3): np.linspace(1, 2, num=10),
        }
    )
    pes._energies.fill(0.0)

    # Cannot plot a PES that has 3 spatial dimensions and one energy in 3D
    with pytest.raises(Exception):
        pes.plot()

    # but can save the energies as a .txt file, which will flatten it
    pes.save(filename="tmp.txt")
    assert np.loadtxt("tmp.txt").shape == (1000,)

    os.remove("tmp.txt")


def test_clear():

    pes = TestPES(species=h2(), rs={(0, 1): (1.5, 10)})
    pes._coordinates = np.ones(
        shape=(10, 2, 3)  # 10 points  # 2 atoms
    )  # x, y, z dimensions

    pes.clear()

    # Clear should reset the energies and zero the coordinates
    for point in pes._points():
        assert np.all(np.isnan(pes._energies[point]))
        assert np.allclose(pes._coordinates[point], 0.0)


def test_point_neighbour():

    pes = TestPES(rs={(0, 1): np.array([1.0, 2.0, 3.0])})

    # A point is not a neighbour if ∆p is zero
    with pytest.raises(ValueError):
        pes._neighbour(point=(1,), dim=0, delta=0)

    # but is if ∆p = +1 for instance
    p = pes._neighbour(point=(1,), dim=0, delta=1)
    assert p == (2,)

    # a point (3,) is not on the surface, as the only valid indices are 0, 1, 2
    p = pes._neighbour(point=(1,), dim=0, delta=2)
    assert not pes._is_contained(p)


def test_spline():
    """Ensure that no matter the ordering of energies a spline can be fit
    to a 2D surface"""

    pes = harmonic_2d_pes()

    _ = pes._spline_2d()

    for i in (0, 1):
        for direction in (1, -1):
            pes._rs[i] = pes._rs[i][::-direction]
            pes._mesh()
            _ = pes._spline_2d()  # Should still be able to spline


def test_relative_energies():

    rel_energies = harmonic_2d_pes().relative_energies

    # Minimum should be 0
    assert np.isclose(np.min(rel_energies), 0.0, atol=1e-6)

    # and support conversion between units
    assert hasattr(rel_energies, "to")


def test_reload_from_only_file():

    harmonic_2d_pes().save("tmp.npz")
    assert os.path.exists("tmp.npz")

    loaded_pes = TestPES.from_file("tmp.npz")
    assert isinstance(loaded_pes, PESnD)
    assert loaded_pes.shape == (21, 21)

    os.remove("tmp.npz")
