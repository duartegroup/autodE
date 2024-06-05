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

        # update the Hessian
        if self.iteration != 0:
            self._coords.update_h_from_old_h(
                self._history.penultimate, self._hessian_update_types
            )
        assert self._coords.h is not None

        n, m = len(self._coords), self._coords.n_constraints
        logger.info(f"Optimising {n} coordinates and {m} lagrange multipliers")

        idxs = self._coords.active_indexes
        n_satisfied_constraints = (n + m - len(idxs)) // 2
        logger.info(
            f"Satisfied {n_satisfied_constraints} constraints. "
            f"Active space is {len(idxs)} dimensional"
        )

        b, u = np.linalg.eigh(self._coords.h)
        constr_idxs = self._get_constraint_idxs_from_eigvecs(u)

        logger.info(
            f"∇^2L has {sum(lmda < 0 for lmda in b)} negative "
            f"eigenvalue(s). Should have {m}"
        )

        logger.debug("Calculating transformed gradient vector")
        f = u.T.dot(self._coords.g)

        delta_s = self._get_rfo_step(b, u, f, constr_idxs)

        # Set all the non-active components of the step to zero
        delta_s[self._coords.inactive_indexes] = 0.0

        self._take_step_within_trust_radius(delta_s)
        return None

    @staticmethod
    def _get_rfo_step(
        b: np.ndarray, u: np.ndarray, f: np.ndarray, uphill_idxs: list
    ):
        """
        Get the partitioned RFO step, uphill along the indices specified
        and downhill along all other indices

        Args:
            b: Eigenvalues of the Hessian
            u: Eigenvectors of the Hessian
            f: Projected gradient along eigenvectors
            uphill_idxs: Indices of eigenvectors along which to take
                    uphill step

        Returns:
            (np.ndarray): The step vector
        """
        m = len(uphill_idxs)
        n = len(b) - len(uphill_idxs)
        delta_s = np.zeros(shape=(m + n,))

        b_max = b[uphill_idxs]
        u_max = u[:, uphill_idxs]
        f_max = f[uphill_idxs]

        b_min = np.delete(b, uphill_idxs)
        u_min = np.delete(u, uphill_idxs, axis=1)
        f_min = np.delete(f, uphill_idxs)

        # form the augmented Hessian for downhill step
        aug_h_min = np.zeros(shape=(n + 1, n + 1))
        aug_h_min[:n, :n] = np.diag(b_min)
        aug_h_min[:-1, -1] = aug_h_min[-1, :-1] = f_min
        lambda_n = np.linalg.eigvalsh(aug_h_min)[0]
        logger.info(f"Calculated λ_p=+{lambda_n:.8f}")
        for i in range(n):
            delta_s -= f_min[i] * u_min[:, i] / (b_min[i] - lambda_n)

        if m == 0:
            return delta_s

        # form the augmented Hessian for uphill step
        aug_h_max = np.zeros(shape=(m + 1, m + 1))
        aug_h_max[:m, :m] = np.diag(b_max)
        aug_h_max[:-1, -1] = aug_h_max[-1, :-1] = f_max
        lambda_p = np.linalg.eigvalsh(aug_h_max)[-1]
        logger.info(f"Calculated λ_p=+{lambda_p:.8f}")
        for i in range(m):
            delta_s -= f_max[i] * u_max[:, i] / (b_max[i] - lambda_p)

        return delta_s

    def _get_constraint_idxs_from_eigvecs(self, u: np.ndarray) -> list:
        """
        Identify which eigenvectors of the Hessian represent the constraint
        mode, since they might not always be in the expected order

        Args:
            u (np.ndarray): The eigenvector matrix with columns representing
                            each eigenvector

        Returns:
            (list[int]): List of indices
        """
        assert self._coords is not None
        m = self._coords.n_constraints
        if m == 0:
            return []

        sum_weights = []
        for idx in range(u.shape[1]):
            # lagrange multiplier weights
            weights = u[:, idx].flatten()[-m:]
            sum_weights.append(np.sum(np.abs(weights)))

        return list(np.argsort(sum_weights)[-m:])

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
