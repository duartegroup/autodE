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
from autode.opt.coordinates.internals import Distances
from autode.opt.coordinates.primitives import ConstrainedDistance, Distance
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
        r"""Partitioned rational function step, seperated along the m
        constrained modes, which need to be maximised, and the remaining
        n-modes which need to be minimised. Modes are in order, such that
        the constrained modes are the first m. Constrained modes must be
        pure distances.

        .. math::

            \mathcal{L} = E(\boldsymbol{R}) - \sum_i^m \lambda_i C_i

        for constraints :math:`C_i` e.g. :math:`R_i - R_i^{req.}`
        """

        self._coords.h = self._updated_h()

        n, m = self._coords.h.shape[0], self._n_constraints
        logger.info('Constructing the Lagrangian second derivative matrix. '
                    f'Have {n} coordinates and {m} Lagrange multiplier(s)')

        # Full set of coordinates are now:  x0, x1, ..., xn, λ0, ... λm

        hess = np.zeros(shape=(n+m, n+m))
        hess[:n, :n] = self._coords.h
        for i in range(m):
            hess[n+i, i] = hess.T[n+i, i] = -1.0

        # Form the gradient in the larger space: ∇L
        g = np.zeros(n+m)
        g[:n] = self._coords.g

        for i in range(m):
            g[i] -= self._lambdas[i]
            g[n+i] = -self._constraints[i].delta(self._coords._x)  # dL/dλ

        b, U = np.linalg.eigh(hess)

        assert sum(b_i < 0 for b_i in b) == m  # Needs m negative eigenvalues

        # Convert the gradient
        F = [np.dot(U[:, i], g) for i in range(n+m)]

        # Set up the step in the diagonal representation
        s_tilde = np.zeros(n+m)

        mat_m = np.zeros(shape=(m+1, m+1))
        mat_m[:m, :m] = np.diag(b[:m])
        mat_m[:-1, m] = mat_m[m, :-1] = F[:m]

        lamd_m, v_m = np.linalg.eigh(mat_m)

        # Largest eigenvalue eigenmode, excluding the final element
        s_tilde[:m] = (v_m[:-1, -1] / v_m[-1, -1])

        mat_n = np.zeros(shape=(n+1, n+1))
        mat_n[:n, :n] = np.diag(b[m:])
        mat_n[:-1, n] = mat_n[n, :-1] = F[m:]

        lamd_n, v_n = np.linalg.eigh(mat_n)
        s_tilde[m:] = (v_n[:-1, 0] / v_n[-1, 0])

        delta_s = np.dot(U, s_tilde)

        self._coords = self._coords + self._sanitised_step(delta_s[:n])
        self._lambdas += delta_s[n:]

        return None

    def _initialise_run(self) -> None:
        """
        Initialise running a partitioned rational function optimisation by
        setting the coordinates, including the constrained distances
        """
        x = CartesianCoordinates(self._species.coordinates)
        primitives = self._generate_primitives(x)

        self._coords = DIC.from_cartesian(x=x, primitives=primitives)
        assert self._constrained_coordinates_are_pure

        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)

        self._update_gradient_and_energy()
        self._lambdas = np.zeros(self._n_constraints)
        return None

    def _generate_primitives(self,
                             x: np.ndarray
                             ) -> 'autode.coordinates.internals.PIC':
        """Primitive internal coordinates from cartesian coordinates"""

        primitives = Distances.from_cartesian(x)

        for (i, j), r in self._species.constraints.distance.items():

            for prim in reversed(primitives):
                if (isinstance(prim, Distance)
                     and sorted((prim.idx_i, prim.idx_j)) == sorted((i, j))):
                    
                    logger.info(f'Primitive: {prim} had overlap with an '
                                f'already present distance. Removing')
                    primitives.remove(prim)

            pic = ConstrainedDistance(idx_i=i, idx_j=j, value=r)
            primitives.append(pic)

        return primitives

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
