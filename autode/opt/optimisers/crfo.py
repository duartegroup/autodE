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

        d2L_eigvals = np.linalg.eigvalsh(self._coords.h)
        logger.info(
            f"∇^2L has {sum(lmda < 0 for lmda in d2L_eigvals)} negative "
            f"eigenvalue(s). Should have {m}"
        )

        # force molecular Hessian block to be positive definite
        hessian = self._coords.h.copy()
        shift = self._coords.rfo_shift
        hessian -= shift * np.eye(n + m)
        for i in range(m):  # no shift on constraints
            hessian[-m + i, -m + i] = 0.0

        logger.info(f"Calculated RFO λ = {shift:.4f}")

        d2L_eigvals = np.linalg.eigvalsh(hessian)
        n_negative = sum(lmda < 0 for lmda in d2L_eigvals)
        if not n_negative == m:
            raise RuntimeError(
                f"Constrained optimisation failed, ∇^2L has {n_negative} "
                f" negative eigenvalues after RFO diagonal shift - "
                f"should have {m}"
            )

        # Set all non-active components of gradient to zero
        gradient = self._coords.g.copy()
        gradient[self._coords.inactive_indexes] = 0.0

        # take a quasi-Newton step
        delta_s = -np.matmul(np.linalg.inv(hessian), gradient)

        # Set all the non-active components of the step to zero
        delta_s[self._coords.inactive_indexes] = 0.0

        self._take_step_within_trust_radius(delta_s)
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
        return None
