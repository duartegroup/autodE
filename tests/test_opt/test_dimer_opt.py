import autode as ade
import numpy as np
from autode.opt.optimisers.dimer import Dimer


class Dimer2D(Dimer):
    """Dimer on a 2D PES that can be easily visualised using the surface

    E = x^2 - y^2

    which generates the classic saddle point:

    __________________
    |       low      |
    |                |
    |high   TS   high|
    |                |
    |       low      |
    ------------------
    """

    def _get_gradient(self, coordinates):
        """E = x^2 - y^2   -->   (dE/dx)_y = 2x  ; (dE/dy)_x = -2y"""
        x, y = coordinates
        return np.array([2.0*x, -2.0*y])

    def __init__(self, x1, x_mid, x2):
        """Initialise with some temporary 2D molecules"""

        self.x1 = x1
        self.x0 = x_mid
        self.x2 = x2

        self.g0 = self._get_gradient(coordinates=self.x0)
        self.g1 = self._get_gradient(coordinates=self.x1)

        self.iterations = DimerIterations()
        self.iterations.append(DimerIteration(phi=0, d=0, dimer=self))


def _test_dimer_2d(plot=False):

    dimer = Dimer2D(x1=np.array([-0.5, -0.5]),
                    x_mid=np.array([-0.25, 0.0]),
                    x2=np.array([0.0, 0.5]))

    # check the distance
    assert np.isclose(dimer.delta, np.sqrt((-0.5)**2 + 1**2)/2.0)

    # optimise the rotation, should be able to be very accurate
    dimer.optimise_rotation(max_iterations=10, phi_tol=1E-2)

    # and check that the rotation does not change the distance between the end
    # points of the dimer
    for iteration in dimer.iterations:
        assert np.isclose(iteration.delta,
                          dimer.iterations[0].delta,
                          atol=1E-1)

    # final iteration should have a change in rotation angle below the
    assert dimer.iterations[-1].phi < 1E-2

    # Do  single translation step
    dimer.translate()

    # then optimise the translation
    while dimer.iterations[-1].d > 1E-2:
        dimer.translate()

    ts_coords = np.zeros(2)  # TS is located at (0, 0) in the (x, y) plane
    assert np.isclose(np.linalg.norm(dimer.x0 - ts_coords), 0, atol=1E-3)

    if plot:
        import matplotlib.pyplot as plt

        x = y = np.arange(-2.0, 2.0, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = X**2 - Y**2

        plt.imshow(Z, extent=[-2, 2, -2, 2])             # show the surface
        plt.scatter([0], [0], marker='o', c='w', s=50)   # mark the TS
        cmap = plt.get_cmap('plasma')

        for i, iteration in enumerate(dimer.iterations):
            x_mid, y_mid = iteration.x0
            x1, y1 = iteration.x1
            x2, y2 = iteration.x2
            plt.plot([x1, x_mid, x2], [y1, y_mid, y2], marker='o',
                     c=cmap(i/10.0))

        plt.tight_layout()
        plt.savefig('dimer_iters', dpi=300)


def _test_dimer_ts_sn2():
    """Test a 'dimer' transition state search for an SN2 reaction"""

    dimer = Dimer(species_1=ade.Molecule('sn2_p1.xyz',
                                         charge=-1, solvent_name='water'),
                  species_2=ade.Molecule('sn2_p2.xyz',
                                         charge=-1, solvent_name='water'),
                  method=ade.methods.XTB())

    dimer.optimise_rotation(phi_tol=0.08, max_iterations=50)

    # Do  single translation step
    dimer.translate()

    # then optimise the translation
    while dimer.iterations[-1].d > 1E-3:
        dimer.translate()

    assert len(dimer.iterations) > 1
    assert dimer.iterations[-1].phi < 0.1

    final_point = ade.Molecule('sn2_p1.xyz')
    final_point.coordinates = dimer.x0
    # TODO: remove this test print
    final_point.print_xyz_file(filename='tmp.xyz')

    dimer.iterations.print_xyz_file(species=dimer._species,
                                    point=0)
