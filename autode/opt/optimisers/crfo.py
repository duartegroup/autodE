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
from autode.exceptions import OptimiserStepError
from autode.opt.optimisers.hessian_update import (
    BFGSDampedUpdate,
    BFGSSR1Update,
)

if TYPE_CHECKING:
    from autode.opt.coordinates.primitives import Primitive

# max and min bounds for the trust radius
_max_trust = 0.2
_min_trust = 0.005


class CRFOptimiser(RFOptimiser):
    """Constrained optimisation in delocalised internal coordinates"""

    def __init__(
        self,
        init_trust: float = 0.1,
        *args,
        extra_prims: Optional[List["Primitive"]] = None,
        trust_update: bool = True,
        max_move: Union[Distance, float] = Distance(0.12, "ang"),
        **kwargs,
    ):
        """
        Constrained rational function optimisation

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Initial value of the trust radius

        Keyword Args:
            extra_prims: A list of aditional coordinates (or constraints) to
                        add to the DIC optimisation space (optional)
            max_move: The maximum distance an atom can move in Cartesian
                    coordinates in a step (assumed units of Å if not given)
            trust_update: Whether to update the trust radius

        See Also:
            :py:meth:`RFOptimiser <RFOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        if not (_min_trust < init_trust < _max_trust):
            init_trust = min(max(init_trust, _min_trust), _max_trust)
            logger.warning(f"Setting trust radius to {init_trust:.3f}")
        self.alpha = float(init_trust)
        self._trust_update = bool(trust_update)
        self._maxmove = Distance(max_move, units="ang")
        assert self._maxmove > 0
        self._extra_prims = [] if extra_prims is None else list(extra_prims)

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
        self._update_trust_radius()

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

        # molecular Hessian block should be positive definite
        hessian = self._coords.h.copy()
        shift = self._coords.rfo_shift
        hessian -= shift * np.eye(n + m)
        for i in range(m):  # no shift on constraints
            hessian[-m + i, -m + i] = 0.0

        logger.info(f"Calculated RFO λ = {shift:.4f}")

        # RFO step in active space
        o = m - n_satisfied_constraints
        hessian = hessian[:, idxs][idxs, :]
        gradient = self._coords.g[idxs]

        d2l_eigvals = np.linalg.eigvalsh(hessian)
        n_negative = sum(lmda < 0 for lmda in d2l_eigvals)
        if n_negative != o:
            raise OptimiserStepError(
                f"Failed to calculated constrained RFO step, ∇^2L has "
                f"{n_negative} negative eigenvalues after RFO diagonal"
                f" shift - should have {o}"
            )

        delta_s_active = -np.matmul(np.linalg.inv(hessian), gradient)

        # form step in full space
        delta_s = np.zeros(shape=(n + m,))
        delta_s[idxs] = delta_s_active

        # scale back to trust radius only on non-constraint modes
        delta_s_q = delta_s[:n]
        if np.linalg.norm(delta_s_q) > self.alpha:
            delta_s = delta_s * self.alpha / np.linalg.norm(delta_s_q)

        self._take_step_within_max_move(delta_s)
        return None

    def _take_step_within_max_move(self, delta_s):
        """
        Take the step by converting internal coordinates to Cartesian
        coordinates, and scaling back if the maximum movement of an
        atom exceeds max_move

        Arguments:
            delta_s: The step in internal coordinates
        """
        assert self._coords is not None

        self._coords.allow_unconverged_back_transform = True
        new_coords = self._coords + delta_s
        cart_delta = new_coords.to("cart") - self._coords.to("cart")
        cart_displ = np.linalg.norm(cart_delta.reshape((-1, 3)), axis=1)
        max_displ = np.abs(cart_displ).max()
        if max_displ > self._maxmove:
            logger.info(
                f"Calculated step too large: max. displacement = "
                f"{max_displ:.3f} Å, scaling down"
            )
            # Note because the transformation is not linear this will not
            # generate a step exactly max(∆x) ≡ α, but is empirically close
            factor = self._maxmove / np.abs(cart_displ).max()
            delta_s = factor * delta_s

        self._coords.allow_unconverged_back_transform = False
        self._coords = self._coords + delta_s

    def _update_trust_radius(self):
        """Updates the trust radius before a geometry step"""
        assert self._coords is not None, "Must have coordinates!"

        if self.iteration == 0:
            return None

        if self._trust_update is False:
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
                self.alpha = min(1.3 * self.alpha, _max_trust)
        elif 1.25 < trust_ratio < 1.75:
            pass
        elif trust_ratio > 1.75:
            self.alpha = max(0.7 * self.alpha, _min_trust)

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
