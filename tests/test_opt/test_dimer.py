import pytest
import numpy as np
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.opt.coordinates.dimer import DimerCoordinates, DimerPoint
from autode.opt.optimisers.dimer import Dimer


def test_dimer_coord_init():

    mol1 = Molecule()
    mol2 = Molecule(atoms=[Atom('H')])

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

    mol1 = Molecule(atoms=[Atom('H'), Atom('H', x=1.0)])
    mol2 = Molecule(atoms=[Atom('H', 0.1), Atom('H', x=1.1)])

    coords = DimerCoordinates.from_species(mol1, mol2)
    assert coords.shape == (3, 6)

    assert np.allclose(coords.x0,
                       np.array([0.05, 0., 0., 1.05, 0., 0.]))

    assert np.allclose(coords.x1,
                       np.array([0., 0., 0., 1., 0., 0.]))

    assert np.allclose(coords.x2,
                       np.array([0.1, 0., 0., 1.1, 0., 0.]))

    # Gradient has not been evaluated
    with pytest.raises(Exception):
        _ = coords.g0


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

        self._coords._set_g_vec(np.array([2.0 * x, -2.0 * y]), point)
        return None

    def _initialise_run(self) -> None:
        """"""
        self._coords.g = np.zeros(shape=(3, 2))

        for point in DimerPoint:
            self._update_gradient_at(point)

        return None


def test_dimer_2d(plot=False):
    arr = np.array([[np.nan, np.nan],   # x0  (midpoint)
                    [-0.5, -0.5],       # x1  (left)
                    [0.0, 0.5]])        # x2  (right)

    dimer = Dimer2D(maxiter=100,
                    coords=DimerCoordinates(arr))

    # Check the midpoint of the dimer is positioned correctly
    assert np.allclose(dimer._coords.x0,
                       np.array([-0.25, 0.0]),
                       atol=1E-10)

    # check the distance between the endf points
    assert np.isclose(dimer._coords.delta,
                      np.sqrt((-0.5) ** 2 + 1 ** 2) / 2.0,
                      atol=1E-10)

    # optimise the rotation, should be able to be very accurate
    dimer._initialise_run()
    dimer._optimise_rotation()

    # and check that the rotation does not change the distance between the end
    # points of the dimer

    for iteration in dimer._history:
        assert np.isclose(iteration.delta,
                          dimer._history[0].delta,
                          atol=1E-1)

    # final iteration should have a change in rotation angle below the
    assert abs(dimer._dc_dphi) < 1E-1

    # Do  single translation step
    dimer._translate()

    # then optimise the translation
    while dimer._history.final.dist > 1E-2:
        dimer._translate()

    # TS is located at (0, 0) in the (x, y) plane
    assert np.allclose(np.linalg.norm(dimer._coords.x0),
                       np.zeros(2),
                       atol=1E-3)

    if plot:
        import matplotlib.pyplot as plt

        x = y = np.arange(-2.0, 2.0, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = X ** 2 - Y ** 2

        plt.imshow(Z, extent=[-2, 2, -2, 2])  # show the surface
        plt.scatter([0], [0], marker='o', c='w', s=50)  # mark the TS
        cmap = plt.get_cmap('plasma')

        for i, iteration in enumerate(dimer.iterations):
            x_mid, y_mid = iteration.x0
            x1, y1 = iteration.x1
            x2, y2 = iteration.x2
            plt.plot([x1, x_mid, x2], [y1, y_mid, y2], marker='o',
                     c=cmap(i / 10.0))

        plt.tight_layout()
        plt.savefig('dimer_iters', dpi=300)


if __name__ == '__main__':
    test_dimer_2d()
