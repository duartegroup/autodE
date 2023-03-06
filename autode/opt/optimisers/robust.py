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

        rfo_h_eff = self._get_rfo_minimise_h_eff()
        rfo_step = -np.linalg.inv(rfo_h_eff) @ self._coords.g
        rfo_step_size = self._get_cart_step_size_from_step(rfo_step)

        if rfo_step_size < self.alpha:
            print(f"Taking a pure RFO step of size: {rfo_step_size:.3f}")
            self._coords = self._coords + rfo_step

        else:
            try:
                qa_h_eff = self._get_qa_minimise_h_eff()
                qa_step = -np.linalg.inv(qa_h_eff) @ self._coords.g
                print(
                    f"Taking a QA step exactly at trust"
                    f" radius: {self.alpha}"
                )
                self._coords = self._coords + qa_step

            except OptimiserStepError:
                # if QA failed, take scaled RFO step
                scaled_step = rfo_step * (self.alpha / rfo_step_size)
                # step size is in Cartesian but scaling should work
                step_size = self._get_cart_step_size_from_step(scaled_step)
                print(
                    f"Taking a scaled RFO step approximately within "
                    f"trust radius: {step_size:.3f}"
                )
                self._coords = self._coords + scaled_step

        return None

    def _get_cart_step_size_from_step(self, step: np.ndarray) -> float:
        """
        Obtains the Cartesian step size given a step in the current
        coordinate system (e.g., Delocalised Internal Coordinates)

        Args:
            step (np.ndarray): The step in the current coordinates

        Returns:
            (float): The step size in Cartesian
        """
        new_coords = self._coords + step
        cart_delta = new_coords.to("cart") - self._coords.to("cart")
        return float(np.linalg.norm(cart_delta))

    def _get_rfo_minimise_h_eff(self) -> np.ndarray:
        """
        Using current Hessian and gradient, obtain the level-shifted
        Hessian that would provide a minimising RFO step

        Returns:
            (np.ndarray): The level-shifted effective Hessian
        """
        h_n = self._coords.h.shape[0]

        # form the augmented Hessian
        aug_h = np.zeros(shape=(h_n + 1, h_n + 1))
        aug_h[:h_n, :h_n] = self._coords.h
        aug_h[-1, :h_n] = self._coords.g
        aug_h[:h_n, -1] = self._coords.g

        aug_h_lmda, aug_h_v = np.linalg.eigh(aug_h)

        # RFO step uses the lowest non-zero eigenvalue
        mode = np.where(np.abs(aug_h_lmda) > 1.0e-15)[0][0]
        lmda = aug_h_lmda[mode]

        # effective hessian = H - lambda * I
        return self._coords.h - lmda * np.eye(h_n)

    def _get_qa_minimise_h_eff(self) -> np.ndarray:
        """
        Using current Hessian and gradient, get the level-shifted Hessian
        for a minimising step, whose magnitude (norm) is equal to the
        trust radius (Quadratic Approximation step).
        Described in J. Golab, D. L. Yeager, Chem. Phys., 78, 1983, 175-199

        Returns:
            (np.ndarray): The level-shifted Hessian for QA step
        """
        # this function is expensive to call
        from scipy.optimize import root_scalar

        h_n = self._coords.h.shape[0]
        h_eigvals = np.linalg.eigvalsh(self._coords.h)
        first_mode = np.where(np.abs(h_eigvals) > 1.0e-15)[0][0]
        first_b = h_eigvals[first_mode]  # first non-zero eigenvalue of H

        def step_length_error(lmda):
            shifted_h = self._coords.h - lmda * np.eye(
                h_n
            )  # level-shifted hessian
            inv_shifted_h = np.linalg.inv(shifted_h)
            step = -inv_shifted_h @ self._coords.g
            step_size = self._get_cart_step_size_from_step(step)
            return step_size - self.alpha

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

        # use the final lambda to construct level-shifted Hessian
        return self._coords.h - res.root * np.eye(h_n)
