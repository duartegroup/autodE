"""Constrained rational function optimisation

Notation follows:
[1] J. Baker, J. Comput. Chem., 18, 8 1080
[2] J. Baker, J. Comput. Chem., 13, 240 Ž1992
"""
import numpy as np

from autode.log import logger
from autode.values import GradientRMS
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.opt.coordinates.internals import InverseDistances
from autode.opt.optimisers.rfo import RFOOptimiser
from autode.opt.optimisers.hessian_update import BFGSUpdate, NullUpdate
from autode.opt.coordinates.primitives import (InverseDistance,
                                               ConstrainedInverseDistance)


class CRFOptimiser(RFOOptimiser):

    def __init__(self,
                 init_alpha:    float = 0.1,
                 *args,
                 **kwargs):
        """
        TODO

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = float(init_alpha)
        self._hessian_update_types = [BFGSUpdate, NullUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""
        self._coords.h = self._updated_h()

        n, m = len(self._coords), self._coords.n_constraints
        logger.info(f"Optimising {n} coordinates and {m} lagrange multipliers")

        idxs = self._coords.active_indexes
        n_satisfied_constraints = (n + m - len(idxs))//2
        logger.info(f"Satisfied {n_satisfied_constraints} constraints. "
                    f"Active space is {len(idxs)} dimensional")

        b, u = np.linalg.eigh(self._coords.h[:, idxs][idxs, :])
        logger.info(f"∇^2L has {sum(lmda < 0 for lmda in b)} negative "
                    f"eigenvalue(s). Should have {m - n_satisfied_constraints}")

        logger.info("Calculating transformed gradient vector")
        f = u.T.dot(self._coords.g[idxs])

        lambda_p = self._lambda_p_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_p={lambda_p:.5f}")

        lambda_n = self._lambda_n_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_n={lambda_n:.5f}")

        # Create the step along the n active DICs and m lagrangian multipliers
        delta_s_active = np.zeros(shape=(len(idxs),))

        o = m - n_satisfied_constraints
        for i in range(o):
            delta_s_active -= f[i] * u[:, i] / (b[i] - lambda_p)

        for j in range(n - n_satisfied_constraints):
            delta_s_active -= f[o+j] * u[:, o+j] / (b[o+j] - lambda_n)

        # Set all the non active components of the step to zero
        delta_s = np.zeros(shape=(n+m,))
        delta_s[idxs] = self._sanitised_step(delta_s_active)

        self._coords = self._coords + delta_s[:n]
        self._coords.set_lagrange_multipliers(delta_s[n:])
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
        self._build_internal_coordinates()
        self._update_hessian_gradient_and_energy()
        return None

    def _build_internal_coordinates(self):
        """Set the initial coordinates to optimise in, formed using
        delocalized internals"""

        if self._species is None:
            raise RuntimeError("Cannot set initial coordinates. No species set")

        if self._coords is None:
            cartesian_coords = CartesianCoordinates(self._species.coordinates)
        else:
            cartesian_coords = self._coords.to("cartesian")
            cartesian_coords.g = None
            cartesian_coords.h = None

        self._coords = DICWithConstraints.from_cartesian(
            x=cartesian_coords,
            primitives=self._primitives
        )
        self._coords.zero_lagrangian_multipliers()
        return None

    @property
    def _primitives(self) -> InverseDistances:
        """Primitive internal coordinates in this molecule"""
        logger.info("Generating primitive internal coordinates")

        pic = InverseDistances()

        for i in range(self._species.n_atoms):
            for j in range(i+1, self._species.n_atoms):

                if (i, j) not in self._species.constraints.distance:
                    pic.append(InverseDistance(i, j))
                    continue

                # i-j is constrained
                r = self._species.constraints.distance[(i, j)]
                pic.append(ConstrainedInverseDistance(i, j, value=1./r))

        logger.info(f"Using {pic.n_constrained} constraints")
        return pic

    def _lambda_p_from_eigvals_and_gradient(self,
                                            b: np.ndarray,
                                            f: np.ndarray) -> float:
        """
        Calculate the positive eigenvalue of the augmented hessian from
        the eigenvalues of the full lagrangian Hessian matrix (b) and the
        gradient in the Hessian eigenbasis
        """
        m = self._coords.n_constraints

        aug_h = np.zeros(shape=(m+1, m+1))
        aug_h[:m, :m] = np.diag(b[:m])
        aug_h[:-1, -1] = aug_h[-1, :-1] = f[:m]

        eigenvalues = np.linalg.eigvalsh(aug_h)
        return eigenvalues[-1]

    def _lambda_n_from_eigvals_and_gradient(self,
                                            b: np.ndarray,
                                            f: np.ndarray) -> float:
        """Like lambda_p but for the negative component"""
        m = self._coords.n_constraints
        n = len(self._coords)

        aug_h2 = np.zeros(shape=(n + 1, n + 1))
        aug_h2[:n, :n] = np.diag(b[m:])
        aug_h2[-1, :n] = aug_h2[:n, -1] = f[m:]

        eigenvalues = np.linalg.eigvalsh(aug_h2)
        return eigenvalues[0]

