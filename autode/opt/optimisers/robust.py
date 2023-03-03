"""
A more robust geometry optimiser, uses features from
multiple optimisation methods. (Similar to those
implemented in common QM packages)

Only minimiser, not a TS search/constrained optimiser
"""
import numpy as np

from autode.opt.optimisers import RFOptimiser
from autode.exceptions import OptimiserStepError


class RobustOptimiser(RFOptimiser):
    def __init__(
        self,
        *args,
        init_trust: float = 0.1,
        min_trust: float = 0.01,
        max_trust: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, init_alpha=init_trust, **kwargs)

        self._max_alpha = float(max_trust)
        self._min_alpha = float(min_trust)

        self._last_pred_e_change = None

    def _step(self) -> None:

        self._coords.h = self._updated_h()
        h_n = self._coords.h.shape[0]

        rfo_lmda = _get_rfo_minimise_lambda(self._coords.h, self._coords.g)
        h_eff = self._coords.h - rfo_lmda * np.eye(h_n)
        rfo_step = -np.linalg.inv(h_eff) @ self._coords.g

        if self._step_scale_factor_within_trust(rfo_step) > 0.99:
            self._coords = self._coords + rfo_step

        else:
            try:
                qa_lmda = _get_qa_minimise_lambda(
                    self._coords.h, self._coords.g, self.alpha
                )
                h_eff = self._coords.h - qa_lmda * np.eye(h_n)
                qa_step = -np.linalg.inv(h_eff) @ self._coords.g
                self._coords = self._coords + qa_step

            except OptimiserStepError:
                # if QA failed, take scaled RFO step
                pass

    def _step_scale_factor_within_trust(self, step: np.ndarray) -> float:
        """
        Returns the scale factor that must be applied to the step
        to have it within the trust radius. If the calculated step
        is smaller than trust radius, return 1.0 i.e. no scaling

        Args:
            step (np.ndarray): The step calculated

        Returns:
            (float): The scale factor
        """
        new_coords = self._coords + step
        cartesian_delta = new_coords.to("cart") - self._coords.to("cart")
        step_size = np.linalg.norm(cartesian_delta)

        # return the scale factor that should be applied
        if step_size <= self.alpha:
            # no scaling if step is smaller
            return 1.0

        elif step_size > self.alpha:
            return self.alpha / step_size


def _get_rfo_minimise_lambda(
    hessian: np.ndarray, gradient: np.ndarray
) -> float:
    """
    Using current Hessian and gradient, obtain the lambda
    (Hessian shift parameter) that would give a minimising
    step, using RFO-like method

    Args:
        hessian (np.ndarray):
        gradient (np.ndarray):
    Returns:
        (float): lambda that gives minimising RFO step
    """
    h_n = hessian.shape[0]

    # form the augmented Hessian
    aug_h = np.zeros(shape=(h_n + 1, h_n + 1), dtype=np.float64)
    aug_h[:h_n, :h_n] = hessian
    aug_h[-1, :h_n] = gradient
    aug_h[:h_n, -1] = gradient

    aug_h_lmda, aug_h_v = np.linalg.eigh(aug_h)

    # RFO step uses the lowest non-zero eigenvalue
    mode = np.where(np.abs(aug_h_lmda) > 1.0e-10)[0][0]
    lmda = aug_h_lmda[mode]

    return lmda


def _get_qa_minimise_lambda(
    hessian: np.ndarray, gradient: np.ndarray, trust: float
) -> float:
    """
    Using current Hessian and gradient, get the lambda (Hessian
    shift parameter) for a minimising step, whose magnitude (norm)
    is equal to the trust radius (Quadratic Approximation step).
    Described in J. Golab, D. L. Yeager, Chem. Phys., 78, 1983, 175-199

    Args:
        hessian (np.ndarray):
        gradient (np.ndarray):
        trust (float):

    Returns:
        (float): lambda that gives the QA minimising step
    """
    from scipy.optimize import root_scalar

    n = hessian.shape[0]
    h_eigvals = np.linalg.eigvalsh(hessian)
    first_mode = np.where(np.abs(h_eigvals) > 1.0e-10)[0][0]
    first_b = h_eigvals[first_mode]  # first non-zero eigenvalue of H

    def step_length_error(lmda):
        shifted_h = hessian - lmda * np.eye(n)  # level-shifted hessian
        inv_shifted_h = np.linalg.inv(shifted_h)
        step = -inv_shifted_h @ gradient.reshape(-1, 1)
        return np.linalg.norm(step) - trust

    # The value of shift parameter lambda must lie within (-infinity, first_b)
    # Need to find the roots of the 1D function step_length_error
    l_plus = 1.0
    for _ in range(1000):
        err = step_length_error(first_b - l_plus)
        if err > 0.0:  # found location where f(x) > 0
            break
        l_plus *= 0.5
    else:  # if loop didn't break
        raise OptimiserStepError("Unable to find lambda where error > 0")

    l_minus = l_plus + 1.0
    for _ in range(1000):
        err = step_length_error(first_b - l_minus)
        if err < 0.0:  # found location where f(x) < 0
            break
        l_minus += 1.0  # reduce lambda by 1.0
    else:
        raise OptimiserStepError("Unable to find lambda where error < 0")

    # Use scipy's root finder
    res = root_scalar(
        step_length_error,
        method="brentq",
        bracket=[first_b - l_minus, first_b - l_plus],
        maxiter=200,
    )

    if not res.converged:
        raise OptimiserStepError("Unable to find root of error function")

    # this is final lambda
    return res.root
