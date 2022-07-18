"""Constrained rational function optimisation

Notation follows:
[1] J. Baker, J. Comput. Chem., 18, 8 1080
[2] J. Baker, J. Comput. Chem., 13, 240 Ž1992
"""
import numpy as np

from itertools import combinations
from autode.log import logger
from autode.values import GradientRMS
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.opt.coordinates.internals import PIC
from autode.opt.optimisers.rfo import RFOptimiser
from autode.opt.optimisers.hessian_update import BFGSPDUpdate, NullUpdate
from autode.opt.coordinates.primitives import (Distance, BondAngle, DihedralAngle,
                                               ConstrainedDistance)


class CRFOptimiser(RFOptimiser):

    def __init__(self,
                 init_alpha: float = 0.05,
                 *args,
                 **kwargs):
        """
        Constrained rational function optimisation

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = float(init_alpha)
        self._hessian_update_types = [BFGSPDUpdate, NullUpdate]

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
        logger.info(f"Calculated λ_p=+{lambda_p:.8f}")

        lambda_n = min(self._lambda_n_from_eigvals_and_gradient(b, f),
                       -1E-3)
        logger.info(f"Calculated λ_n={lambda_n:.8f}")

        # Create the step along the n active DICs and m lagrangian multipliers
        delta_s_active = np.zeros(shape=(len(idxs),))

        o = m - n_satisfied_constraints
        logger.info(f"Maximising {o} modes")
        for i in range(o):
            delta_s_active -= f[i] * u[:, i] / (b[i] - lambda_p)

        for j in range(n - n_satisfied_constraints):
            delta_s_active -= f[o+j] * u[:, o+j] / (b[o+j] - lambda_n)

        # Set all the non-active components of the step to zero
        delta_s = np.zeros(shape=(n+m,))
        delta_s[idxs] = delta_s_active

        self._take_sanitised_step(delta_s[:n])
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
        logger.info("Initialising constrained optimisation")

        self._build_internal_coordinates()
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()

        return None

    def _build_internal_coordinates(self):
        """Set the initial coordinates to optimise in, formed using
        delocalized internals"""

        if self._species is None:
            raise RuntimeError("Cannot set initial coordinates. No species set")

        cartesian_coords = CartesianCoordinates(self._species.coordinates)
        self._coords = DICWithConstraints.from_cartesian(
            x=cartesian_coords,
            primitives=self._primitives
        )
        self._coords.zero_lagrangian_multipliers()
        return None

    @property
    def _primitives(self) -> PIC:
        """Primitive internal coordinates in this molecule"""
        logger.info("Generating primitive internal coordinates")
        graph = self._species.graph

        pic = PIC()

        for (i, j) in sorted(graph.edges):

            if (not self._species.constraints.distance
                    or (i, j) not in self._species.constraints.distance):
                pic.append(Distance(i, j))
                continue

            # i-j is constrained
            r = self._species.constraints.distance[(i, j)]
            pic.append(ConstrainedDistance(i, j, value=r))

        for o in range(self._species.n_atoms):
            for (n, m) in combinations(graph.neighbors(o), r=2):
                angle = BondAngle(o=o, m=m, n=n)

                if angle not in pic:
                    pic.append(angle)

        if self._species.n_atoms > 2 and self._species.is_planar():
            for (o, p) in graph.edges:
                for m in graph.neighbors(o):
                    if m == p:
                        continue

                    # TODO: test angle is not 180º

                    for n in graph.neighbors(p):
                        if n == o:
                            continue

                        pic.append(DihedralAngle(m, o, p, n))

        logger.info(f"Using {pic.n_constrained} constraints in {len(pic)} "
                    f"primitive internal coordinates")
        return pic

    def _lambda_p_from_eigvals_and_gradient(self,
                                            b: np.ndarray,
                                            f: np.ndarray) -> float:
        """
        Calculate the positive eigenvalue of the augmented hessian from
        the eigenvalues of the full lagrangian Hessian matrix (b) and the
        gradient in the Hessian eigenbasis
        """
        m = self._coords.n_constraints - self._coords.n_satisfied_constraints

        aug_h = np.zeros(shape=(m+1, m+1))
        aug_h[:m, :m] = np.diag(b[:m])
        aug_h[:-1, -1] = aug_h[-1, :-1] = f[:m]

        eigenvalues = np.linalg.eigvalsh(aug_h)
        return eigenvalues[-1]

    def _lambda_n_from_eigvals_and_gradient(self,
                                            b: np.ndarray,
                                            f: np.ndarray) -> float:
        """Like lambda_p but for the negative component"""
        n_satisfied = self._coords.n_satisfied_constraints
        m = self._coords.n_constraints - n_satisfied
        n = len(self._coords) - n_satisfied

        aug_h = np.zeros(shape=(n + 1, n + 1))
        aug_h[:n, :n] = np.diag(b[m:])
        aug_h[-1, :n] = aug_h[:n, -1] = f[m:]

        eigenvalues = np.linalg.eigvalsh(aug_h)
        return eigenvalues[0]

