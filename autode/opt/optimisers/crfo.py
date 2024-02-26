"""
Constrained rational function optimisation

Notation follows:
[1] J. Baker, J. Comput. Chem., 18, 8 1080
[2] J. Baker, J. Comput. Chem., 13, 240 Ž1992
"""
import numpy as np
from typing import Union

from autode.log import logger
from autode.values import GradientRMS, Distance
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.opt.coordinates.internals import AnyPIC
from autode.opt.optimisers.rfo import RFOptimiser
from autode.opt.optimisers.hessian_update import (
    BFGSDampedUpdate,
    BFGSSR1Update,
)


class CRFOptimiser(RFOptimiser):
    def __init__(
        self, init_alpha: Union[Distance, float] = 0.05, *args, **kwargs
    ):
        """
        Constrained rational function optimisation

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size, assumed Angstrom if units
                        not given

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = Distance(init_alpha, units="ang")
        assert self.alpha > 0
        self._hessian_update_types = [BFGSDampedUpdate, BFGSSR1Update]

    def _step(self) -> None:
        """Partitioned rational function step"""
        assert self._coords is not None, "Must have coords to take a step"
        assert self._coords.g is not None, "Must have a gradient"
        self._coords.h = self._updated_h()

        n, m = len(self._coords), self._coords.n_constraints
        logger.info(f"Optimising {n} coordinates and {m} lagrange multipliers")

        idxs = self._coords.active_indexes
        n_satisfied_constraints = (n + m - len(idxs)) // 2
        logger.info(
            f"Satisfied {n_satisfied_constraints} constraints. "
            f"Active space is {len(idxs)} dimensional"
        )

        b, u = np.linalg.eigh(self._coords.h[:, idxs][idxs, :])
        logger.info(
            f"∇^2L has {sum(lmda < 0 for lmda in b)} negative "
            f"eigenvalue(s). Should have {m - n_satisfied_constraints}"
        )

        logger.info("Calculating transformed gradient vector")
        f = u.T.dot(self._coords.g[idxs])

        lambda_p = self._lambda_p_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_p=+{lambda_p:.8f}")

        lambda_n = self._lambda_n_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_n={lambda_n:.8f}")

        # Create the step along the n active DICs and m lagrangian multipliers
        delta_s_active = np.zeros(shape=(len(idxs),))

        o = m - n_satisfied_constraints
        logger.info(f"Maximising {o} modes")

        for i in range(o):
            delta_s_active -= f[i] * u[:, i] / (b[i] - lambda_p)

        for j in range(n - n_satisfied_constraints):
            delta_s_active -= f[o + j] * u[:, o + j] / (b[o + j] - lambda_n)

        # Set all the non-active components of the step to zero
        delta_s = np.zeros(shape=(n + m,))
        delta_s[idxs] = delta_s_active

        c = self._take_step_within_trust_radius(delta_s[:n])
        self._coords.update_lagrange_multipliers(c * delta_s[n:])
        return None

    @property
    def _g_norm(self) -> GradientRMS:
        """Calculate the norm of the gradient in the active subspace"""

        if self._coords is None or self._coords.g is None:
            return super()._g_norm

        gradient = self._coords.g[self._coords.active_indexes]
        return GradientRMS(np.sqrt(np.mean(np.square(gradient))))

    def _initialise_run(self) -> None:
        """Initialise the optimisation"""
        logger.info("Initialising optimisation")

        self._build_internal_coordinates()
        assert self._coords is not None
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()

        return None

    def _build_internal_coordinates(self):
        """Set the initial coordinates to optimise in, formed using
        delocalized internals"""

        if self._species is None:
            raise RuntimeError(
                "Cannot set initial coordinates. No species set"
            )

        cartesian_coords = CartesianCoordinates(self._species.coordinates)
        primitives = AnyPIC.from_species(self._species)

        self._coords = DICWithConstraints.from_cartesian(
            x=cartesian_coords, primitives=primitives
        )
        self._coords.zero_lagrangian_multipliers()
        return None

    def _lambda_p_from_eigvals_and_gradient(
        self, b: np.ndarray, f: np.ndarray
    ) -> float:
        """
        Calculate the positive eigenvalue of the augmented hessian from
        the eigenvalues of the full lagrangian Hessian matrix (b) and the
        gradient in the Hessian eigenbasis
        """
        assert self._coords is not None

        m = self._coords.n_constraints - self._coords.n_satisfied_constraints

        aug_h = np.zeros(shape=(m + 1, m + 1))
        aug_h[:m, :m] = np.diag(b[:m])
        aug_h[:-1, -1] = aug_h[-1, :-1] = f[:m]

        eigenvalues = np.linalg.eigvalsh(aug_h)
        return eigenvalues[-1]

    def _lambda_n_from_eigvals_and_gradient(
        self, b: np.ndarray, f: np.ndarray
    ) -> float:
        """Like lambda_p but for the negative component"""
        assert self._coords is not None

        n_satisfied = self._coords.n_satisfied_constraints
        m = self._coords.n_constraints - n_satisfied
        n = len(self._coords) - n_satisfied

        aug_h = np.zeros(shape=(n + 1, n + 1))
        aug_h[:n, :n] = np.diag(b[m:])
        aug_h[-1, :n] = aug_h[:n, -1] = f[m:]

        eigenvalues = np.linalg.eigvalsh(aug_h)
        return eigenvalues[0]
