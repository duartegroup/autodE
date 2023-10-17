"""
Dimer method for finding transition states given two points on the PES.
Notation follows
1. https://aip.scitation.org/doi/10.1063/1.2815812
based on
2. https://aip.scitation.org/doi/10.1063/1.2104507
3. https://aip.scitation.org/doi/10.1063/1.480097

-------------------------------------------------------
x : Cartesian coordinates
g : gradient in cartesian coordinates
"""
import numpy as np

from typing import Optional, Union, TYPE_CHECKING
from enum import Enum

from autode.calculations import Calculation
from autode.log import logger
from autode.values import GradientRMS, Angle, MWDistance
from autode.opt.optimisers.base import Optimiser
from autode.opt.coordinates.dimer import DimerCoordinates, DimerPoint


if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class Dimer(Optimiser):
    """Dimer spanning two points on the PES with a TS at the midpoint"""

    def __init__(
        self,
        maxiter: int,
        coords: DimerCoordinates,
        ratio_rot_iters: int = 10,
        gtol: Union[float, GradientRMS] = GradientRMS(1e-3, "Ha Å-1"),
        trns_tol: MWDistance = MWDistance(1e-3, "Å amu^1/2"),
        phi_tol: Angle = Angle(5.0, "°"),
        init_alpha: MWDistance = MWDistance(0.3, "Å amu^1/2"),
    ):
        """
        Dimer optimiser

        -----------------------------------------------------------------------
        Arguments:
            maxiter: Maximum number of gradient evaluations to do

            coords: Coordinates of the dimer, consisting of the end points and
                    the interpolated midpoint

            ratio_rot_iters: Number of rotations per translation in each
                             dimer step

            gtol: Tolerance on the gradient at the midpoint for convergence

            trns_tol: Tolerance on the minimum root mean square translation
                      distance, below which convergence is signaled

            phi_tol: Tolerance on the rotation angle below which rotation is
                     not performed

            init_alpha: Initial step size to use in mass-weighted cartesian
                        coordinates
        """
        super().__init__(maxiter=maxiter, coords=coords)

        self._ratio_rot_iters = ratio_rot_iters
        self.gtol = gtol
        self.trns_tol = trns_tol
        self.phi_tol = phi_tol
        self.init_alpha = init_alpha

        self._converged_translation = False  # Convergence flag

        assert self._coords is not None
        logger.info(f"Initialised a dimer with Δ = {self._coords.delta:.4f} Å")

    @classmethod
    def optimise(
        cls,
        species: "Species",
        method: "Method",
        n_cores: Optional[int] = None,
        coords: Optional[DimerCoordinates] = None,
        **kwargs,
    ) -> None:
        """
        Optimise a dimer pair of coordinates such that the species coordinates
        are close to a transition state

        -----------------------------------------------------------------------
        Arguments:
            species: Species to optimise to a TS using dimer iterations

            method: Electronic structure method to use to optimise

            n_cores: Number of cores to use for the optimisation

            coords: Dimer coordinates
        """

        if not isinstance(coords, DimerCoordinates):
            raise ValueError(
                "A dimer optimisation must be initialised from "
                "a set of dimer coordinates"
            )

        optimiser = cls(maxiter=100, coords=coords)
        optimiser.run(species=species, method=method, n_cores=n_cores)
        return None

    def _step(self) -> None:
        """
        Do a single dimer optimisation step, consisting of several rotation and
        translation steps.
        """
        self._update_gradient_at(DimerPoint.left)

        self._optimise_rotation()
        trns_result = self._translate()

        if trns_result == _StepResult.skipped_translation:
            self._converged_translation = True

        return None

    def _rotate(self) -> "_StepResult":
        """Apply a rotation"""
        assert self._coords is not None

        c_phi0, dc_dphi0 = self._c, self._dc_dphi
        logger.info(
            "Doing a single dimer rotation to minimise the curvature. "
            f"Current: C = {c_phi0:.4f} "
            f"and dC/dϕ = {dc_dphi0:.4f}"
        )

        cached_coordinates = self._coords.copy()

        phi_1 = self._phi1.to("radians")
        logger.info(
            f'Rotating by ϕ = {phi_1.to("degrees"):.4f}º and '
            f"evaluating the curvature"
        )

        if abs(phi_1) < self.phi_tol:
            logger.info("Rotation angle was below the threshold, not rotating")
            return _StepResult.skipped_rotation

        self._rotate_coords(phi_1, update_g1=True)

        b1 = 0.5 * dc_dphi0  # eqn. 8 from ref. [1]

        a1 = (
            c_phi0 - self._c + b1 * np.sin(2 * phi_1)
        ) / (  # eqn. 9 from ref. [1]
            1 - 2.0 * np.cos(2.0 * phi_1)
        )

        a0 = 2.0 * (c_phi0 - a1)  # eqn. 10 from ref. [1]
        phi_min = Angle(0.5 * np.arctan(b1 / a1), units="radians")
        logger.info(f'ϕ_min = {phi_min.to("degrees"):.4f}º')

        c_min = (
            0.5 * a0 + a1 * np.cos(2.0 * phi_min) + b1 * np.sin(2.0 * phi_min)
        )

        if c_min > c_phi0:
            logger.info(
                "Optimised curvature was larger than the initial, "
                "adding π/2"
            )
            phi_min += np.pi / 2.0

        # Rotate back from the test point, then to the minimum
        self._coords = cached_coordinates
        self._rotate_coords(phi=phi_min, update_g1=True)

        return _StepResult.did_rotation

    def _translate(self, update_g0=True) -> "_StepResult":
        """Translate the dimer under the translational force"""
        assert self._coords is not None
        x0 = self._coords.x0

        trns_iters = [c for c in self._history if c.did_translation]

        if len(trns_iters) < 2:
            step_size = float(self.init_alpha)
            logger.info(
                f"Did not have two previous translation step, guessing"
                f" α = {step_size} Å"
            )

        else:
            prev_trns_iter = trns_iters[-2]
            logger.info(
                f"Did {len(trns_iters)} previous translations, can "
                f"calculate α using the Barzilai–Borwein method"
            )

            step_size = (
                np.abs(
                    np.dot(
                        (x0 - prev_trns_iter.x0),
                        (self._coords.f_t - prev_trns_iter.f_t),
                    )
                )
                / np.linalg.norm(self._coords.f_t - prev_trns_iter.f_t) ** 2
            )

        delta_x = step_size * self._coords.f_t
        trns_rms = MWDistance(np.sqrt(np.mean(np.square(delta_x))))

        if trns_rms < self.trns_tol:
            logger.info(f"Step length small than tolerance {self.trns_tol}")
            return _StepResult.skipped_translation

        logger.info(f"Translating by ~{trns_rms:.4f} per coordinate")

        coords = self._coords.copy()
        coords += delta_x

        self._coords = coords
        self._coords.phi = Angle(0.0)  # Did not rotation
        self._coords.dist = trns_rms  # but did translate

        if update_g0:
            self._update_gradient_at(DimerPoint.midpoint)

        return _StepResult.did_translation

    def _initialise_run(self) -> None:
        """Initialise running the dimer optimisation"""
        assert self._coords is not None and self._species is not None

        if np.isclose(self._coords.delta, 0.0):
            raise RuntimeError("Zero distance between the dimer points")

        self._coords._g = np.zeros(shape=(3, 3 * self._species.n_atoms))

        # TODO: Hessian. Ref [1] shows that a BFGS step to the translation
        # and rotation -> faster convergence than SD steps

        self._update_gradient_at(DimerPoint.midpoint)
        self._update_gradient_at(DimerPoint.left)

        return None

    @property
    def converged(self) -> bool:
        """Has the dimer converged?"""
        assert self._coords is not None

        if self._converged_translation:
            logger.info(
                "Converged purely based on translation of the "
                "dimer midpoint"
            )
            return True

        rms_g0 = np.sqrt(np.mean(np.square(self._coords.g0)))
        return self.iteration > 0 and rms_g0 < self.gtol

    def _update_gradient_and_energy(self) -> None:
        """Update the gradient at the midpoint"""
        return self._update_gradient_at(DimerPoint.midpoint)

    def _update_gradient_at(self, point: DimerPoint) -> None:
        """Update the gradient at one of the points in the dimer"""
        assert (
            self._coords is not None
            and self._species is not None
            and self._method is not None
            and self._method.keywords.grad is not None
        )
        i = int(point)

        self._species.coordinates = self._coords.x_at(
            point, mass_weighted=False
        )

        calc = Calculation(
            name=f"{self._species.name}_{i}_{self.iteration}",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.grad,
            n_cores=self._n_cores,
        )
        calc.run()
        assert (
            self._species.energy is not None
            and self._species.gradient is not None
        )

        self._coords.e = self._species.energy
        self._coords.set_g_at(
            point, self._species.gradient.flatten(), mass_weighted=False
        )

        calc.clean_up(force=True, everything=True)
        return None

    @property
    def _theta(self) -> np.ndarray:
        """Optimisation direction"""
        return self._theta_steepest_descent

    @property
    def _theta_steepest_descent(self) -> np.ndarray:
        """Rotation direction Θ, calculated using steepest descent"""
        assert self._coords is not None

        f_r = self._coords.f_r
        # F_R / |F_R| with a small jitter to prevent division by zero
        return f_r / (np.linalg.norm(f_r) + 1e-8)

    @property
    def _c(self) -> float:
        """Curvature of the PES, C_τ.  eqn. 4 in ref [1]"""
        assert self._coords is not None

        g1, g0 = self._coords.g1, self._coords.g0
        return np.dot((g1 - g0), self._coords.tau_hat) / self._coords.delta

    @property
    def _dc_dphi(self) -> float:
        """dC_τ/dϕ eqn. 6 in ref [1]"""
        assert self._coords is not None

        g1, g0 = self._coords.g1, self._coords.g0

        return 2.0 * np.dot((g1 - g0), self._theta) / self._coords.delta

    @property
    def _phi1(self) -> Angle:
        """φ_1. eqn 5 in ref [1]"""
        val = -0.5 * np.arctan(self._dc_dphi / (2.0 * np.linalg.norm(self._c)))
        return Angle(val, "radians")

    def _rotate_coords(self, phi: Angle, update_g1: bool = True) -> None:
        """
        Rotate the dimer by an angle phi around the midpoint.
        eqn. 13 in ref. [2]

        Arguments:
            phi (float): Rotation angle in radians (ϕ)

            update_g1 (bool): Update the gradient on point 1 after the rotation
        """
        assert self._coords is not None

        x0 = self._coords.x0.copy()  # Midpoint coordinates
        g0 = self._coords.g0.copy()  # Midpoint gradient

        delta = self._coords.delta * (
            self._coords.tau_hat * np.cos(phi.to("rad"))
            + self._theta * np.sin(phi.to("rad"))
        )

        max_step_c = np.max(np.abs(self._coords.x1.copy() - (x0 + delta)))

        if max_step_c > self.init_alpha:
            logger.warning(
                f"Step size ({max_step_c}) was above the tolerance"
                f" {self.init_alpha} Å amu^1/2. Scaling down"
            )
            return self._rotate_coords(phi=Angle(phi / 2), update_g1=update_g1)

        self._coords = self._coords.copy()
        self._coords.x1 = x0 + delta
        self._coords.x2 = x0 - delta

        self._coords.dist = MWDistance(0.0)
        self._coords.phi = phi

        # Midpoint has not moved so it's gradient its retained
        self._coords.g0 = g0

        # But both the end points have, so clear their gradients
        self._coords.g1[:] = self._coords.g2[:] = np.nan

        if update_g1:
            self._update_gradient_at(DimerPoint.left)

        logger.info(
            f"Rotated coordinates, now have |g1 - g0| = "
            f"{np.linalg.norm(self._coords.g1 - self._coords.g0):.4f}."
            f" ∆ = {self._coords.delta:.3f}"
        )
        return None

    def _optimise_rotation(self):
        """Rotate the dimer optimally"""
        logger.info(
            f"Minimising dimer rotation up to "
            f'δϕ = {self.phi_tol.to("degrees"):.4f}º'
        )

        for i in range(self._ratio_rot_iters):
            result = self._rotate()

            if (
                result == _StepResult.skipped_rotation
                or abs(self._coords.phi) < self.phi_tol
            ):
                break

            logger.info(
                f"Micro iteration: {i}."
                f' ϕ={self._coords.phi.to("degrees"):.2f}º'
            )

        return None


class _StepResult(Enum):
    did_rotation = 0
    skipped_rotation = 1

    did_translation = 2
    skipped_translation = 3
