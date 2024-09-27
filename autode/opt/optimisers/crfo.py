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
MAX_TRUST = 0.2
MIN_TRUST = 0.01


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

        if not (MIN_TRUST < init_trust < MAX_TRUST):
            init_trust = min(max(init_trust, MIN_TRUST), MAX_TRUST)
            logger.warning(f"Setting trust radius to {init_trust:.3f}")

        self.alpha = float(init_trust)
        self._trust_update = bool(trust_update)
        self._maxmove = Distance(max_move, units="ang")
        assert self._maxmove > 0
        self._extra_prims = [] if extra_prims is None else list(extra_prims)

        self._hessian_update_types = [BFGSDampedUpdate, BFGSSR1Update]

    def _log_constrained_opt_progress(self):
        """Log information about the constraints"""
        n, m = len(self._coords), self._coords.n_constraints
        s = self._coords.n_satisfied_constraints
        logger.info(f"Optimising {n} coordinates and {m} lagrange multipliers")

        idxs = self._coords.active_indexes
        logger.info(
            f"Satisfied {s} constraints. Active space"
            f" is {len(idxs)} dimensional"
        )
        d2l_ev = np.linalg.eigvalsh(self._coords.h[:, idxs][idxs, :])
        logger.info(
            f"Hessian in active space has {sum(k < 0 for k in d2l_ev)} "
            f"negative eigenvalue(s). Should have {m-s}"
        )
        return None

    def _step(self) -> None:
        """Partitioned rational function step"""
        assert self._coords is not None, "Must have coords to take a step"

        if self.iteration != 0:
            self._coords.update_h_from_old_h(
                self._history.penultimate, self._hessian_update_types
            )
        assert self._coords.h is not None

        self._update_trust_radius()
        self._log_constrained_opt_progress()

        # get RFO step
        delta_s = self._get_rfo_step()

        # scale back to trust radius only on non-constraint modes
        n = len(self._coords)
        delta_s_q = delta_s[:n]
        if np.linalg.norm(delta_s_q) > self.alpha:
            logger.info("Scaling RFO step to trust radius")
            delta_s = delta_s * self.alpha / np.linalg.norm(delta_s_q)

        logger.info("Taking an RFO step")
        self._take_step_within_max_move(delta_s)
        return None

    def _get_rfo_step(self):
        """
        Calculate the unscaled RFO step, for the correct set of
        coordinates

        Returns:
            (np.ndarray): The RFO step
        """
        n, m = len(self._coords), self._coords.n_constraints
        idxs = self._coords.active_indexes

        # only molec. Hessian should be +ve definite
        lmda = self._coords.rfo_shift
        hess = self._coords.h - lmda * np.eye(n + m)
        # no shift on constraints
        for i in range(m):
            hess[-m + i, -m + i] = 0.0

        logger.info(f"Calculated RFO λ = {lmda:.4f}")
        # RFO step in active space
        hess = hess[:, idxs][idxs, :]
        grad = self._coords.g[idxs]
        self._check_shifted_hessian_has_correct_struct(hess)
        full_step = np.zeros(shape=(n + m))
        rfo_step = -np.matmul(np.linalg.inv(hess), grad)
        full_step[idxs] = rfo_step

        return full_step

    def _check_shifted_hessian_has_correct_struct(self, arr) -> None:
        """
        Check that the shifted Hessian from RFO or QA has correct
        eigenvalue structure

        Args:
            arr (np.ndarray): Shifted hessian to check

        Raises:
            (OptimiserStepError): if Hessian does not have correct structure
        """
        assert self._coords is not None
        m = self._coords.n_constraints
        o = m - self._coords.n_satisfied_constraints
        ev = np.linalg.eigvalsh(arr)
        n_negative = sum(k < 0 for k in ev)
        if not o == n_negative:
            raise OptimiserStepError(
                f"Failed to obtain step, shifted Hessian should have {o}"
                f" negative eigenvalue(s), but has {n_negative}"
            )
        return None

    def _take_step_within_max_move(self, delta_s: np.ndarray):
        """
        Take the step by converting internal coordinates to Cartesian
        coordinates, and scaling back if the maximum movement of an
        atom exceeds max_move

        Arguments:
            delta_s (np.ndarray): The step in internal coordinates
        """
        assert self._coords is not None

        self._coords.allow_unconverged_back_transform = True
        new_coords = self._coords + delta_s
        cart_delta = new_coords.to("cart") - self._coords.to("cart")
        cart_displ = np.linalg.norm(cart_delta.reshape((-1, 3)), axis=1)
        max_displ = np.abs(cart_displ).max()

        self._coords.allow_unconverged_back_transform = False
        if max_displ > self._maxmove:
            logger.info(
                f"Calculated step too large: max. displacement = "
                f"{max_displ:.3f} Å, scaling down"
            )
            # Note because the transformation is not linear this will not
            # generate a step exactly max(∆x) ≡ α, but is empirically close
            factor = self._maxmove / max_displ
            self._coords = self._coords + (factor * delta_s)

        else:
            self._coords = new_coords

        return None

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
            self.alpha = max(0.7 * self.alpha, MIN_TRUST)
        elif 0.25 < trust_ratio < 0.75:
            pass
        elif 0.75 < trust_ratio < 1.25:
            # increase if step was actually near trust radius
            if abs(last_step_size - self.alpha) / self.alpha < 0.05:
                self.alpha = min(1.3 * self.alpha, MAX_TRUST)
        elif 1.25 < trust_ratio < 1.75:
            pass
        elif trust_ratio > 1.75:
            self.alpha = max(0.7 * self.alpha, MIN_TRUST)

        logger.info(
            f"Ratio of actual/predicted dE = {trust_ratio:.3f},"
            f" Current trust radius = {self.alpha:.3f}"
        )

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
