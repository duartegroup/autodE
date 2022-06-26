"""Constrained rational function optimisation

Notation follows:
[1] J. Baker, J. Comput. Chem., 18, 8 1080
[2] J. Baker, J. Comput. Chem., 13, 240 Ž1992
"""
import numpy as np

from autode.log import logger
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.opt.coordinates.internals import InverseDistances
from autode.opt.optimisers.rfo import RFOOptimiser
from autode.opt.optimisers.hessian_update import BFGSUpdate, NullUpdate
from autode.opt.coordinates.primitives import (InverseDistance,
                                               ConstrainedInverseDistance)


class CRFOptimiser(RFOOptimiser):

    def __init__(self,
                 init_alpha:    float = 0.02,
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
        self._hessian_update_types = [NullUpdate]# , BFGSUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""
        print(np.linalg.norm(self._coords._x[:3] - self._coords._x[3:6]))

        self._coords.h = self._updated_h()

        if self._coords.some_constraints_are_satisfied:
            logger.info("Rebuilding DIC without satisfied primitives")
            g = self._coords.to("cartesian").reshape((-1, 3))
            e = self._coords.e
            raise NotImplementedError("Can't rebuild coordinates because "
                                      "there is not now all the weight along"
                                      "a constrained mode. need to be smarter")
            self._build_internal_coordinates()
            self._remove_lagrange_multipliers()
            self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
            self._coords.make_hessian_positive_definite()
            self._coords.update_g_from_cart_g(g)
            self._coords.e = e

        n, m = len(self._coords), self._coords.n_constraints

        b, u = np.linalg.eigh(self._coords.h)
        logger.info(f"∇^2L has {sum(lmda < 0 for lmda in b)} negative "
                    f"eigenvalue(s). Should have {m}")

        logger.info("Calculating transformed gradient vector")
        f = u.T.dot(self._coords.g)

        lambda_p = self._lambda_p_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_p={lambda_p:.5f}")

        lambda_n = self._lambda_n_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_n={lambda_n:.5f}")

        # Create the step along the n active DICs and m lagrangian multipliers
        delta_s = np.zeros(shape=(n+m,))

        for i in range(m):
            delta_s -= f[i] * u[:, i] / (b[i] - lambda_p)

        for j in range(n):
            delta_s -= f[m+j] * u[:, m+j] / (b[m+j] - lambda_n)

        self._coords._lambda[:] = self._sanitised_step(delta_s[n:])
        self._coords = self._coords + self._sanitised_step(delta_s[:n])
        return None

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

        cartesian_coords = CartesianCoordinates(self._species.coordinates)

        self._coords = DICWithConstraints.from_cartesian(
            x=cartesian_coords,
            primitives=self._primitives
        )
        self._coords.zero_lagrangian_multipliers()
        return None

    @property
    def _primitives(self) -> InverseDistances:
        """Primitive internal coordinates in this molecule"""

        pic = InverseDistances()

        if self._coords is not None:
            x = self._coords.to("cartesian")
        else:
            x = CartesianCoordinates(self._species.coordinates)

        for i in range(self._species.n_atoms):
            for j in range(i+1, self._species.n_atoms):

                if (i, j) not in self._species.constraints.distance:
                    pic.append(InverseDistance(i, j))
                    continue

                # i-j is constrained
                r = self._species.constraints.distance[(i, j)]
                p = ConstrainedInverseDistance(i, j, value=1./r)

                if p.is_satisfied(x):
                    logger.info(f"Not including constrained primitive: {i}-{j}")
                    continue

                pic.append(p)

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

    def _remove_lagrange_multipliers(self) -> None:
        """
        Remove the lagrange multipliers associated with constrained primitives
        that are satisfied from the coordinate space. This also requires
        setting the components of the gradient and Hessian having removed
        some coordinates from the optimisation space (along with their
        associated lagrangian multipliers for e.g. R-R_0). Gradient vector has
        the form:

        g = (g_0, g_1, ... g_n-1, g_n, dL/dλ_0, dL/dλ_1)

        where, for example, if the first constraint is satisfied then g_n-1
        and dL/dλ_0 are removed.
        """
        s = self._history.penultimate

        idxs = s.satisfied_constraint_indexes
        assert len(idxs) > 0  # the previous coordinates must have >0 satisfied
        logger.info(f"Removing {len(idxs)} Lagrange multipliers & associated "
                    f"satisfied primitives from optimisation space")

        n, m = len(s), s.n_constraints

        _lambda = []
        for i, lambda_i in enumerate(s._lambda):
            if i not in idxs:
                _lambda.append(i)

        self._coords._lambda = np.array(_lambda)

        def idx_should_be_removed(_i: int) -> bool:
            if any(_i == _j + n for _j in idxs):
                logger.info(f"Component {_i} was a Lagrange multiplier")
                return True

            if any(_i == n - 1 - _j for _j in idxs):
                logger.info(f"Component {_i} was a satisfied primitive")
                return True

            return False

        print(np.round(s.g, 3))
        print(np.round(s.h, 3))
        print()

        n_new, m_new = len(self._coords), self._coords.n_constraints
        g = np.zeros(shape=(n_new+m_new))
        h = np.zeros(shape=(n_new+m_new, n_new+m_new))

        i_new = 0
        for i in range(n + m):

            if not idx_should_be_removed(i):
                g[i_new] = s.g[i]

                j_new = 0
                for j in range(n + m):
                    print(i_new, j_new)
                    if not idx_should_be_removed(j):
                        h[i_new, j_new] = s.h[i, j]
                        j_new += 1

                i_new += 1

        self._coords.e = s.e  # Energy is unchanged
        self._coords.g = g
        print(np.round(g, 3))

        self._coords.h = h
        print(np.round(h, 3))


        return None
