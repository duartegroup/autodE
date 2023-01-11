import pytest
import numpy as np
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.methods import XTB
from autode.values import MWDistance
from autode.opt.coordinates.dimer import DimerCoordinates, DimerPoint
from autode.opt.optimisers.dimer import Dimer
from .molecules import methane_mol
from ..testutils import requires_with_working_xtb_install


def _single_atom_dimer_coords():
    return DimerCoordinates(np.arange(9, dtype=float).reshape(3, 3))


def test_dimer_coord_init():

    # Dimer coordinates must be a 3xn matrix of the mid and two end points
    with pytest.raises(ValueError):
        _ = DimerCoordinates(np.array([0.0, 0.1]))

    with pytest.raises(ValueError):
        _ = DimerCoordinates(np.arange(2).reshape(2, 2))

    coords = _single_atom_dimer_coords()
    assert not coords.did_rotation
    assert not coords.did_translation

    assert coords != "a"


def test_dimer_coord_mol_init():

    mol1 = Molecule()
    mol2 = Molecule(atoms=[Atom("H")], mult=2)

    # Dimer coordinates must be created from two species with the same
    # atomic composition
    with pytest.raises(ValueError):
        _ = DimerCoordinates.from_species(mol1, mol2)

    # Dimer coordinates are concatenated cartesian coordinates
    coords = DimerCoordinates.from_species(mol2, mol2)
    assert coords.shape == (3, 3)

    assert coords.g is None
    assert coords.h is None


def test_dimer_coord_init_polyatomic():

    mol1 = Molecule(atoms=[Atom("H"), Atom("H", x=1.0)])
    mol2 = Molecule(atoms=[Atom("H", 0.1), Atom("H", x=1.1)])

    coords = DimerCoordinates.from_species(mol1, mol2)
    assert coords.shape == (3, 6)

    # Coordinates are mass weighted, so not precisely the below values
    assert np.allclose(
        coords.x0, np.array([0.05, 0.0, 0.0, 1.05, 0.0, 0.0]), atol=0.1
    )

    assert np.allclose(
        coords.x1, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), atol=0.1
    )

    assert np.allclose(
        coords.x2, np.array([0.1, 0.0, 0.0, 1.1, 0.0, 0.0]), atol=0.1
    )

    # Gradient has not been evaluated
    with pytest.raises(Exception):
        _ = coords.g0


def test_dimer_invalid_update():

    coords = _single_atom_dimer_coords()

    # Cannot update the gradient from an individual cartesian gradient, as the
    # point on the dimer must be specified
    with pytest.raises(Exception):
        coords.update_g_from_cart_g(np.zeros(3))

    # Currently there is no Hessian to be updated with, so anything goes
    coords.update_h_from_cart_h(None)
    assert coords.h is None

    # Not a valid conversion dimer to individual cartesian
    with pytest.raises(Exception):
        _ = coords.to("cartesian")


def test_repr():

    assert "dimer" in repr(_single_atom_dimer_coords()).lower()


def test_mass_weighting_no_masses():

    coords = _single_atom_dimer_coords()

    with pytest.raises(Exception):
        _ = coords.x_at(DimerPoint.midpoint, mass_weighted=False)

    with pytest.raises(Exception):
        _ = coords.set_g_at(
            DimerPoint.midpoint, np.zeros(3), mass_weighted=False
        )


def test_dimer_init_zero_distance():

    a = Molecule(atoms=[Atom("H")], mult=2)

    dimer = Dimer(maxiter=10, coords=DimerCoordinates.from_species(a, a))

    # Should raise an exception if there is no distance between the end points
    with pytest.raises(RuntimeError):
        dimer._initialise_run()


def test_dimer_coords_phi_set():

    coords = _single_atom_dimer_coords()

    # Phi must be an angle with defined units
    with pytest.raises(ValueError):
        coords.phi = "a"

    with pytest.raises(ValueError):
        coords.phi = 0.0


class Dimer2D(Dimer):
    r"""
    Dimer on a 2D PES

    E = x^2 - y^2

    which generates the classic saddle point::

        __________________
        |       low      |
        |                |
        |high   TS   high|
        |                |
        |       low      |
        ------------------
    """

    __test__ = False

    def _update_gradient_at(self, point) -> None:
        """E = x^2 - y^2   -->   (dE/dx)_y = 2x  ; (dE/dy)_x = -2y"""
        if point == DimerPoint.midpoint:
            x, y = self._coords.x0
        else:
            x, y = self._coords[int(point), :]

        self._coords.set_g_at(point, np.array([2.0 * x, -2.0 * y]))
        return None

    def _initialise_run(self) -> None:
        self._coords.g = np.zeros(shape=(3, 2))

        for point in DimerPoint:
            self._update_gradient_at(point)

        return None


def test_dimer_2d():
    arr = np.array(
        [
            [np.nan, np.nan],
            [-0.5, -0.5],
            [0.0, 0.5],
        ]  # x0  (midpoint)  # x1  (left)
    )  # x2  (right)

    dimer = Dimer2D(
        maxiter=100, coords=DimerCoordinates(arr), init_alpha=MWDistance(0.5)
    )

    # Check the midpoint of the dimer is positioned correctly
    assert np.allclose(dimer._coords.x0, np.array([-0.25, 0.0]), atol=1e-10)

    # check the distance between the endf points
    assert np.isclose(
        dimer._coords.delta, np.sqrt((-0.5) ** 2 + 1**2) / 2.0, atol=1e-10
    )

    # optimise the rotation, should be able to be very accurate
    dimer._initialise_run()
    dimer._optimise_rotation()

    # and check that the rotation does not change the distance between the end
    # points of the dimer

    for iteration in dimer._history:
        assert np.isclose(iteration.delta, dimer._history[0].delta, atol=1e-1)

    # final iteration should have a change in rotation angle below the
    assert abs(dimer._dc_dphi) < 0.2

    # Do  single translation step
    dimer._translate()

    # then optimise the translation
    while dimer._history.final.dist > 1e-2:
        dimer._translate()

    # TS is located at (0, 0) in the (x, y) plane
    assert np.allclose(
        np.linalg.norm(dimer._coords.x0), np.zeros(2), atol=1e-3
    )


@requires_with_working_xtb_install
def test_dimer_sn2():

    left_point = Molecule(
        name="sn2_left",
        charge=-1,
        mult=1,
        solvent_name="water",
        atoms=[
            Atom("F", -5.09333, 4.39680, 0.09816),
            Atom("Cl", -0.96781, 4.55705, -0.06369),
            Atom("C", -3.36921, 4.46281, 0.03034),
            Atom("H", -3.24797, 3.87380, -0.85688),
            Atom("H", -3.17970, 4.00237, 0.97991),
            Atom("H", -3.27773, 5.52866, -0.04735),
        ],
    )

    right_point = Molecule(
        name="sn2_right",
        charge=-1,
        mult=1,
        solvent_name="water",
        atoms=[
            Atom("F", -5.38530, 4.38519, 0.10942),
            Atom("Cl", -0.93502, 4.55732, -0.06547),
            Atom("C", -3.05723, 4.47536, 0.01839),
            Atom("H", -3.26452, 3.87898, -0.84787),
            Atom("H", -3.19779, 4.00624, 0.97190),
            Atom("H", -3.29590, 5.51839, -0.04586),
        ],
    )

    coords = DimerCoordinates.from_species(left_point, right_point)
    dimer = Dimer(maxiter=50, coords=coords, ratio_rot_iters=10)

    ts = left_point.new_species("ts")
    dimer.run(species=ts, method=XTB(), n_cores=1)

    ts.coordinates = dimer._history.final.x_at(
        DimerPoint.midpoint, mass_weighted=False
    )
    ts.print_xyz_file(filename="tmp.xyz")

    assert dimer.iteration > 1
    # assert dimer._history.final.phi.to('degrees') < 10
    assert dimer.converged


def test_dimer_optimise_no_coordinates():

    # Cannot use a dimer optimiser on a single species, unlike other optimisers
    with pytest.raises(Exception):

        Dimer.optimise(species=methane_mol(), method=XTB())
