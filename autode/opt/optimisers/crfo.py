"""
Constrained rational function optimisation

Notation follows:
[1] J. Baker, J. Comput. Chem., 18, 8 1080
[2] J. Baker, J. Comput. Chem., 13, 240 Ž1992
"""
import numpy as np
from typing import Union, Optional, List, TYPE_CHECKING

from autode.log import logger
from autode.values import GradientRMS, Distance
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.opt.coordinates.internals import AnyPIC
from autode.opt.optimisers.rfo import RFOptimiser
from autode.opt.optimisers.hessian_update import (
    BFGSDampedUpdate,
    BFGSSR1Update,
)

if TYPE_CHECKING:
    from autode.opt.coordinates.primitives import Primitive

# max and min bounds for the trust radius
_max_trust = 0.3
_min_trust = 0.005


class CRFOptimiser(RFOptimiser):
    """Constrained optimisation in delocalised internal coordinates"""

    def __init__(
        self,
        init_alpha: Union[Distance, float] = 0.05,
        *args,
        extra_prims: Optional[List["Primitive"]] = None,
        **kwargs,
    ):
        """
        Constrained rational function optimisation

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size, assumed Angstrom if units
                        not given

        Keyword Args:
            extra_prims: A list of aditional coordinates (or constraints) to
                        add to the DIC optimisation space (optional)

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = Distance(init_alpha, units="ang")
        assert self.alpha > 0
        self._hessian_update_types = [BFGSDampedUpdate, BFGSSR1Update]

        if extra_prims is None:
            extra_prims = []
        self._extra_prims = list(extra_prims)

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

        d2l_eigvals = np.linalg.eigvalsh(self._coords.h)
        logger.info(
            f"∇^2L has {sum(lmda < 0 for lmda in d2l_eigvals)} negative "
            f"eigenvalue(s). Should have {m}"
        )

        # force molecular Hessian block to be positive definite
        hessian = self._coords.h.copy()
        shift = self._coords.rfo_shift
        hessian -= shift * np.eye(n + m)
        for i in range(m):  # no shift on constraints
            hessian[-m + i, -m + i] = 0.0

        logger.info(f"Calculated RFO λ = {shift:.4f}")

        d2l_eigvals = np.linalg.eigvalsh(hessian)
        n_negative = sum(lmda < 0 for lmda in d2l_eigvals)
        if not n_negative == m:
            raise RuntimeError(
                f"Constrained optimisation failed, ∇^2L has {n_negative} "
                f" negative eigenvalues after RFO diagonal shift - "
                f"should have {m}"
            )

        # take quasi-Newton step in active subspace
        hessian = hessian[:, idxs][idxs, :]
        gradient = self._coords.g[idxs]
        delta_s_active = -np.matmul(np.linalg.inv(hessian), gradient)

        # form step in full space
        delta_s = np.zeros(shape=(n + m,))
        delta_s[idxs] = delta_s_active

        self._take_step_within_trust_radius(delta_s)
        return None

    def _update_trust_radius(self):
        """Updates the trust radius before a geometry step"""
        assert self._coords is not None, "Must have coordinates!"

        if self.iteration == 0:
            return None

        coords_l = self._history.penultimate
        pred_delta_e = coords_l.pred_quad_delta_e(self._coords)
        trust_ratio = self.last_energy_change / float(pred_delta_e)
        last_step_size = np.linalg.norm(
            np.array(coords_l) - np.array(self._coords)
        )

        if trust_ratio < 0.25:
            self.alpha = max(0.7 * self.alpha, _min_trust)
        elif 0.25 < trust_ratio < 0.75:
            pass
        elif 0.75 < trust_ratio < 1.25:
            # increase if step was actually near trust radius
            if abs(last_step_size - self.alpha) / self.alpha < 0.05:
                self.alpha = min(1.2 * self.alpha, _max_trust)
        elif 1.25 < trust_ratio < 1.75:
            pass
        elif trust_ratio > 1.75:
            self.alpha = max(0.9 * self.alpha, _min_trust)

        logger.info(
            f"Ratio of actual/predicted dE = {trust_ratio:.3f},"
            f" Current trust radius = {self.alpha:.3f}"
        )

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
        for ic in self._extra_prims:
            primitives.add(ic)

        self._coords = DICWithConstraints.from_cartesian(
            x=cartesian_coords, primitives=primitives
        )
        return None
