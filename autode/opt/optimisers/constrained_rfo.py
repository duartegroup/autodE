"""
Perform a rational function optimisation in delocalised internal coordinates
(DIC) to a minimum under a set of distance constraints. Requires:

1. Forming a set of DIC that contains the distances as primitives (as well)
   as the other completely redundant set
2. Schmidt-orthogonalsing the resultant U matrix
3. Remove primitives coordinates with zero weight
4. Form the Lagrangian function
5. Modify the Hessian matrix with

Refs.

[1] J. Baker, A. Kessi and B. Delley, J. Chem. Phys. 105, 1996, 192
[2] J. Baker, J. Comput. Chem. 18, 1997, 1079
[3] https://manual.q-chem.com/5.2/A1.S5.html
"""
import numpy as np
from typing import Optional, List
from autode.log import logger
from autode.opt.coordinates.internals import InverseDistances
from autode.opt.coordinates.primitives import ConstrainedDistance
from autode.opt.coordinates import CartesianCoordinates, DIC
from autode.opt.optimisers.rfo import RFOOptimiser
from autode.opt.optimisers.hessian_update import BFGSUpdate, NullUpdate


class CRFOOptimiser(RFOOptimiser):

    def __init__(self,
                 init_alpha:    float = 0.1,
                 *args,
                 **kwargs):
        """
        Constrained rational function optimisation

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Initial step size

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)
        self.alpha = float(init_alpha)

        self._lambdas: Optional[np.ndarray] = None
        self._hessian_update_types = [BFGSUpdate, NullUpdate]

    def _step(self) -> None:
        """Partitioned rational function step, seperated along the m
        constrained modes, which need to be maximised, and the remaining
        n-modes which need to be minimised. Modes are in order, such that
        the constrained modes are the first m.
        """

        self._coords.h = self._updated_h()

        n, m = self._coords.h.shape[0], self._n_constraints
        logger.info('Constructing the Lagrangian second derivative matrix. '
                    f'Have {n} coordinates and {m} Lagrange multiplier(s)')

        print(np.round(self._coords.U, 2))
        print(self._coords.primitives)
        print()

        hess = np.zeros(shape=(n+m, n+m))
        hess[:n, :n] = self._coords.h
        for i in range(m):
            hess[n+i, i] = hess.T[n+i, i] = -1.0

        print(np.round(hess, 2))

        # Determine the eigenvalues and eigenvectors of the extended Hessian
        b, u = np.linalg.eigh(hess)

        print('EIGENVALS:')
        print(b)
        print()

        # Form the gradient in the larger space
        g = np.zeros(n+m)
        g[:n] = self._coords.g

        print(self._constraints[0].delta(self._coords._x))
        for i in range(m):
            g[i] += self._lambdas[i]
            g[n+i] = -self._constraints[i].delta(self._coords._x)

        # Gradient of the coordinates and constraints in the eigenbasis
        g_tilde = np.dot(u.T, g)

        # Set up the total step
        s_tilde = np.zeros_like(g_tilde)

        aug_m = np.diag(np.concatenate((b[:m], np.zeros(1))))
        aug_m[:-1, -1] = g_tilde[:m]
        aug_m[-1, :-1] = g_tilde[:m]

        _, _v = np.linalg.eigh(aug_m)
        s_tilde[:m] = _v[:-1, -1] / _v[-1, -1]  # Largest eigenvalue

        aug_n = np.diag(np.concatenate((b[m:], np.zeros(1))))
        aug_n[:-1, -1] = g_tilde[m:]
        aug_n[-1, :-1] = g_tilde[m:]

        _, _v = np.linalg.eigh(aug_n)
        s_tilde[m:] = _v[:-1, 0] / _v[-1, 0]

        delta_s = np.dot(u, s_tilde)

        print(delta_s)

        self._coords = self._coords + self._sanitised_step(delta_s[:n])
        self._lambdas -= delta_s[n:]

        return None

    def _initialise_run(self) -> None:
        """
        Initialise running a partitioned rational function optimisation by
        setting the coordinates, including the constrained distances
        """
        x = CartesianCoordinates(self._species.coordinates)

        primitives = InverseDistances.from_cartesian(x)

        for (i, j), r in self._species.constraints.distance.items():
            pic = ConstrainedDistance(idx_i=i, idx_j=j, value=r)

            # WARNING: other functions expect constraints to be at the front
            primitives.insert(0, pic)

        self._coords = DIC.from_cartesian(x, primitives=primitives)
        assert self._constrained_coordinates_are_pure

        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)

        self._update_gradient_and_energy()
        self._lambdas = np.zeros(self._n_constraints)
        return None

    def _remove_satisfied_coordinates(self) -> None:
        raise NotImplementedError

    @property
    def _constraints(self) -> List[ConstrainedDistance]:
        """Constraints applied to this set of coordinates"""
        return [c for c in self._coords.primitives
                if isinstance(c, ConstrainedDistance)]

    @property
    def _n_constraints(self) -> int:
        return len(self._species.constraints.distance)

    @property
    def _constrained_coordinates_are_pure(self) -> bool:
        """
        Are the constrained coordinates 'pure' in the delocalised set?
        For example, one constrained distance should lead to the first DIC
        being comprised of exactly one primitive (that distance), while for
        m constrained distances there should be m pure coordinates
        """

        def _is_pure(vec):
            return (sum(np.isclose(x, 1.0) for x in vec) == 1
                    and sum(np.isclose(x, 0.0) for x in vec) == len(vec) - 1)

        matrix = self._coords.U

        return all(_is_pure(matrix[:, i]) for i in range(self._n_constraints))
