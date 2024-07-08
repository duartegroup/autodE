"""
Constrained optimisation with trust radius model
"""
import numpy as np
from scipy.optimize import root_scalar
from typing import Union, Optional, List, TYPE_CHECKING

from autode.log import logger
from autode.values import GradientRMS, Distance
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.opt.optimisers.hessian_update import BFGSPDUpdate, BFGSSR1Update
from autode.exceptions import OptimiserStepError

if TYPE_CHECKING:
    from autode.opt.coordinates.primitives import Primitive


class TRMOptimiser(CRFOptimiser):
    """Trust-radius model optimiser in delocalised internal coordinates"""

    def __init__(
        self,
        init_alpha: Union[Distance, float] = 0.1,
        *args,
        extra_prims: Optional[List["Primitive"]] = None,
        **kwargs,
    ):
        """
        Constrained trust-radius model optimisation

        -----------------------------------------------------------------------

        Args:
            init_alpha:
            *args:
            extra_prims:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.alpha = Distance(init_alpha, units="ang")
        assert self.alpha > 0

        self._hessian_update_types = [BFGSPDUpdate, BFGSSR1Update]

        if extra_prims is None:
            extra_prims = []
        self._extra_prims = list(extra_prims)

    def _step(self) -> None:
        """Trust radius step"""
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

        def get_trm_step(hess, grad, lmda):
            """TRM step from hessian, gradient and shift"""
            hess = hess - lmda * np.eye(hess.shape[0])
            for i in range(m):  # no shift on constraints
                hess[-m + i, -m + i] = 0.0
            full_step = np.zeros_like(grad)
            hess = hess[:, idxs][idxs, :]
            grad = grad[idxs]
            trm_step = -np.matmul(np.linalg.inv(hess), grad)
            full_step[idxs] = trm_step
            return full_step

        def trm_step_error(lmda):
            """Get the TRM step size - trust radius for lambda value"""
            ds = get_trm_step(self._coords.h, self._coords.g, lmda)
            ds_atoms = ds[:n]
            return np.linalg.norm(ds_atoms) - self.alpha

        min_b = self._coords.min_eigval
        # try simple quasi-Newton if hessian is positive definite
        if min_b > 0 and trm_step_error(0.0) < 0.0:
            step = get_trm_step(self._coords.h, self._coords.g, 0.0)
            self._take_step_within_trust_radius(step)
            return None

        # next try RFO step if within trust radius
        rfo_shift = self._coords.rfo_shift
        rfo_step = get_trm_step(self._coords.h, self._coords.g, rfo_shift)
        if np.linalg.norm(rfo_step) <= self.alpha:
            logger.info(f"Calculated RFO λ = {rfo_shift}")
            self._take_step_within_trust_radius(rfo_step)
            return None

        # constrain step to trust radius
        try:
            d = 1.0
            # bisection to find upper bound
            for _ in range(500):
                right_bound = min_b - d
                if trm_step_error(right_bound) > 0:
                    break
                d = d / 2
            assert trm_step_error(right_bound) > 0

            left_bound = right_bound - 0.1
            for _ in range(1000):
                if trm_step_error(left_bound) < 0:
                    break
                left_bound -= 0.1
            if trm_step_error(left_bound) > 0:
                raise OptimiserStepError(
                    "Unable to find bounds for root search"
                )

            res = root_scalar(
                f=trm_step_error, bracket=[left_bound, right_bound]
            )
            if not res.converged:
                raise OptimiserStepError("Root search did not converge")
            logger.info(f"Calculated TRM λ = {res.root:.4f}")
            step = get_trm_step(self._coords.h, self._coords.g, res.root)
            self._take_step_within_trust_radius(step)
            return None

        # TRM failed, switch to scaled RFO
        except OptimiserStepError as exc:
            logger.info(
                f"TRM step failed: {str(exc)}, switching to scaled RFO"
            )
            logger.info(f"Calculated RFO λ = {rfo_shift}")
            # use scaled RFO
            rfo_step = rfo_step * self.alpha / np.linalg.norm(rfo_step)
            self._take_step_within_trust_radius(rfo_step)
            return None
