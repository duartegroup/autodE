import numpy as np
import pytest

from autode.species import Molecule
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.trust_region import (
    CauchyTROptimiser,
    DoglegTROptimiser,
    CGSteihaugTROptimiser,
)
from .setup import Method


def branin_energy(x, y):
    return (y - 0.129 * x**2 + 1.6 * x - 6) ** 2 + 6.07 * np.cos(x) + 10


class BraninCauchyTROptimiser(CauchyTROptimiser):

    __test__ = False

    def _space_has_degrees_of_freedom(self) -> bool:
        return True

    def _update_gradient_and_energy(self) -> None:
        """Update the gradient and energy for the Branin function

        f(x, y) = (y - 0.129x^2 + 1.6x - 6)^2 + 6.07cos(x) + 10
        """
        x, y = self._coords

        self._coords.e = branin_energy(x, y)

        grad = [
            (
                2 * (1.6 - 0.258 * x) * (y - 0.129 * x**2 + 1.6 * x - 6)
                - 6.07 * np.sin(x)
            ),
            (2 * (y - 0.129 * x**2 + 1.6 * x - 6)),
        ]

        self._coords.g = np.array(grad)

        h_xx = (
            2 * (1.6 - 0.258 * x) ** 2
            - 0.516 * (-0.129 * x**2 + 1.6 * x + y - 6)
            - 6.07 * np.cos(x)
        )
        h_xy = 2 * (1.6 - 0.258 * x)
        h_yy = 2
        self._coords.h = np.array([[h_xx, h_xy], [h_xy, h_yy]])

    def _log_convergence(self) -> None:

        attrs = [
            self.iteration,
            self._coords.e,
            *self._coords,
            self.rho,
            self.alpha,
            np.linalg.norm(self.p),
            self._g_norm,
        ]

        if hasattr(self, "tau"):
            attrs.append(self.tau)

        for thing in attrs:
            print(f"{round(thing, 3):10.3f}" f"", end=" ")
        print()

    @property
    def converged(self) -> bool:
        return np.linalg.norm(self._coords.g) < self._gtol


class BraninDoglegTROptimiser(DoglegTROptimiser):
    def _space_has_degrees_of_freedom(self) -> bool:
        return True

    def _update_gradient_and_energy(self) -> None:
        return BraninCauchyTROptimiser._update_gradient_and_energy(self)

    def _log_convergence(self) -> None:
        return BraninCauchyTROptimiser._log_convergence(self)

    @property
    def converged(self) -> bool:
        return np.linalg.norm(self._coords.g) < self._gtol


class BraninCGSteihaugTROptimiser(CGSteihaugTROptimiser):
    def _space_has_degrees_of_freedom(self) -> bool:
        return True

    def _update_gradient_and_energy(self) -> None:
        return BraninCauchyTROptimiser._update_gradient_and_energy(self)

    def _log_convergence(self) -> None:
        return BraninCauchyTROptimiser._log_convergence(self)

    @property
    def converged(self) -> bool:
        return np.linalg.norm(self._coords.g) < self._gtol


def test_trm_base_properties():
    init_coords = CartesianCoordinates([6.0, 14.0])

    optimiser = BraninCauchyTROptimiser(
        maxiter=20,
        etol=100,  # Some large value
        trust_radius=2.0,
        coords=init_coords,
        gtol=0.01,
    )

    # Updating the Hessian should be possible, and be the identity matrix
    CauchyTROptimiser._update_hessian(optimiser)
    assert np.allclose(optimiser._coords.h, np.eye(len(init_coords)))

    # rho requires a gradient to be evaluated
    optimiser._history.append(CartesianCoordinates([5.5, 13.0]))
    with pytest.raises(Exception):
        _ = optimiser.rho


def test_branin_minimisation():
    """Uses the example from:
    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
    """

    init_coords = CartesianCoordinates([6.0, 14.0])

    optimiser = BraninCauchyTROptimiser(
        maxiter=20,
        etol=100,  # Some large value
        trust_radius=2.0,
        coords=init_coords,
        gtol=0.01,
        max_trust_radius=5.0,
        t_1=0.25,
        t_2=2.0,
        eta_1=0.2,
        eta_2=0.25,
        eta_3=0.75,
    )

    optimiser.run(Molecule(name="blank"), method=Method())
    assert optimiser.converged
    assert np.allclose(optimiser._coords, np.array([3.138, 2.252]), atol=0.01)


def test_branin_dogleg_minimisation():

    # Dogleg optimiser doesn't seem to be so efficient - TODO: Check
    optimiser = BraninDoglegTROptimiser(
        maxiter=1000,
        etol=100,  # Some large value
        trust_radius=2.0,
        coords=CartesianCoordinates([6.0, 14.0]),
        gtol=0.01,
        max_trust_radius=5.0,
        t_1=0.25,
        t_2=2.0,
        eta_1=0.2,
        eta_2=0.25,
        eta_3=0.75,
    )

    optimiser.run(Molecule(name="blank"), method=Method())

    assert optimiser.converged
    assert np.allclose(optimiser._coords, np.array([3.138, 2.252]), atol=0.02)


def test_branin_cg_minimisation():

    optimiser = BraninCGSteihaugTROptimiser(
        maxiter=1000,
        etol=100,  # Some large value
        trust_radius=2.0,
        coords=CartesianCoordinates([6.0, 14.0]),
        gtol=0.01,
        max_trust_radius=5.0,
        t_1=0.25,
        t_2=2.0,
        eta_1=0.2,
        eta_2=0.25,
        eta_3=0.75,
    )

    optimiser.run(Molecule(name="blank"), method=Method())

    assert optimiser.converged
    assert np.allclose(optimiser._coords, np.array([3.138, 2.252]), atol=0.02)

    # Should also be able to optimise directly
    coords = CartesianCoordinates([6.0, 14.0])
    BraninCGSteihaugTROptimiser.optimise(
        Molecule(name="blank"),
        method=Method(),
        gtol=0.01,
        etol=10,
        coords=coords,
    )

    assert optimiser.converged


def test_base_cg_properties():

    optimiser = CGSteihaugTROptimiser(
        maxiter=10,
        trust_radius=1.0,
        etol=1,
        gtol=0.1,
        coords=CartesianCoordinates([0.1, 0.0]),
    )

    assert not optimiser.converged

    optimiser._coords.g = np.zeros(shape=(2,))
    optimiser._solve_subproblem()

    # For an almost zero gradient the step size should be zero
    assert np.linalg.norm(optimiser.p) < 1e-10
