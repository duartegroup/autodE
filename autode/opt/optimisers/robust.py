"""
A more robust geometry optimiser, uses features from
multiple optimisation methods. (Similar to those
implemented in common QM packages)

Only minimiser, not a TS search/constrained optimiser
"""
import numpy as np

from autode.opt.coordinates import CartesianCoordinates, DIC
from autode.opt.coordinates.primitives import Distance
from autode.opt.optimisers import CRFOptimiser
from autode.opt.optimisers.hessian_update import FlowchartUpdate
from autode.exceptions import OptimiserStepError
from autode.values import GradientRMS
from autode.log import logger
from itertools import combinations


class RobustOptimiser(CRFOptimiser):
    def __init__(
        self,
        *args,
        coord_type: str = "dic",
        init_trust: float = 0.1,
        update_trust: bool = True,
        min_trust: float = 0.01,
        max_trust: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, init_alpha=init_trust, **kwargs)

        self._max_alpha = float(max_trust)
        self._min_alpha = float(min_trust)
        self._upd_alpha = bool(update_trust)

        if coord_type.lower() in ["cart", "cartesian"]:
            self._coord_type = "cart"
        elif coord_type.lower() in ["dic"]:
            self._coord_type = "dic"
        else:
            raise ValueError("Coordinate type must be either 'cart' or 'dic'")
        # todo stop if any constraints detected
        # todo damp if required
        self._last_damp_iter = None

        self._hessian_update_types = [FlowchartUpdate]

    @property
    def converged(self) -> bool:
        if self._species is not None and self._species.n_atoms == 1:
            return True  # Optimisation 0 DOF is always converged

        # both gradient and energy must be converged!!
        return self._abs_delta_e < self.etol and self._g_norm < self.gtol

    def _initialise_run(self) -> None:
        self._build_coordinates()
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()

    def _build_coordinates(self) -> None:
        """Initialise the optimisation"""
        cart_coords = CartesianCoordinates(self._species.coordinates)
        if self._coord_type == "cart":
            self._coords = cart_coords

        else:
            primitives = self._primitives
            if len(primitives) < cart_coords.expected_number_of_dof:
                logger.info(
                    "Had an incomplete set of primitives. Adding "
                    "additional distances"
                )
                for i, j in combinations(range(self._species.n_atoms), 2):
                    primitives.append(Distance(i, j))
            self._coords = DIC.from_cartesian(
                x=cart_coords, primitives=primitives
            )

        return None

    @property
    def _g_norm(self) -> GradientRMS:
        """Calculate the RMS gradient norm in Cartesian coordinates"""
        grad = self._coords.to("cart").g
        return GradientRMS(np.sqrt(np.average(np.square(grad))))

    def _step(self) -> None:

        self._coords.h = self._updated_h()
        self._update_trust_radius()

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
        for a minimising step, whose magnitude (norm) is approximately
        equal to the trust radius (Quadratic Approximation step).

        Described in J. Golab, D. L. Yeager, and P. Jorgensen,
        Chem. Phys., 78, 1983, 175-199

        Returns:
            (np.ndarray): The level-shifted Hessian for QA step
        """
        # this function is expensive to call
        from scipy.optimize import root_scalar

        h_n = self._coords.h.shape[0]
        h_eigvals = np.linalg.eigvalsh(self._coords.h)
        first_mode = np.where(np.abs(h_eigvals) > 1.0e-15)[0][0]
        first_b = h_eigvals[first_mode]  # first non-zero eigenvalue of H

        def get_int_step_size_and_deriv(lmda):
            """Get the step, step size and the derivative
            for the given lambda"""
            shifted_h = self._coords.h - lmda * np.eye(h_n)
            inv_shifted_h = np.linalg.inv(shifted_h)
            step = -inv_shifted_h @ self._coords.g
            size = np.linalg.norm(step)
            deriv = -np.linalg.multi_dot((step, inv_shifted_h, step))
            deriv = float(deriv) / size
            return size, deriv, step

        def optimise_lambda_for_int_step(int_size, lmda_guess=None):
            """Given a step length in internal coords, get the
            value of lambda that will give that step"""
            if lmda_guess is None:
                lmda_guess = first_b - 1.0
            for _ in range(20):
                size, der, _ = get_int_step_size_and_deriv(lmda_guess)
                if abs(size - int_size) / int_size < 0.001:
                    break
                lmda_guess -= (1 - size / int_size) * (size / der)
                print("size=", size, "deriv=", der, "lambda=", lmda_guess)
            else:
                raise OptimiserStepError("Failed in optimising internal step")
            return lmda_guess

        last_lmda = None

        def cart_step_length_error(int_size):
            nonlocal last_lmda
            last_lmda = optimise_lambda_for_int_step(int_size, last_lmda)
            _, _, step = get_int_step_size_and_deriv(last_lmda)
            step_size = self._get_cart_step_size_from_step(step)
            return step_size - self.alpha

        # The value of shift parameter lambda must lie within (-infinity, first_b)
        # Need to find the roots of the 1D function cart_step_length_error
        int_step_size = self.alpha  # starting guess, so ok to use cartesian
        fac = 1.0
        size_min_bound = None
        size_max_bound = None

        for _ in range(10):
            int_step_size = int_step_size * fac
            err = cart_step_length_error(int_step_size)
            if err < 0:  # found where error is < 0
                fac = 2.0  # increase the step size
                size_min_bound = int_step_size
            else:  # found where error is >= 0
                fac = 0.5  # decrease the step size
                size_max_bound = int_step_size
            if (size_max_bound is not None) and (size_min_bound is not None):
                break
        else:
            raise OptimiserStepError(
                "Unable to find bracket range for root finding"
            )
        print("Starting Brent's procedure to find root")
        # Use scipy's root finder
        res = root_scalar(
            cart_step_length_error,
            method="brentq",
            bracket=[size_min_bound, size_max_bound],
            maxiter=20,
            rtol=0.01,  # 1% margin of error
        )
        print("obtained int step size = ", res.root)

        if not res.converged:
            raise OptimiserStepError("Unable to find optimal lambda for step")

        final_lmda = optimise_lambda_for_int_step(res.root)

        print("final lambda=", final_lmda, "last lambda = ", last_lmda)
        # use the final lambda to construct level-shifted Hessian
        return self._coords.h - final_lmda * np.eye(h_n)

    def _update_trust_radius(self):
        """
        Updates the trust radius before a geometry step
        """
        # skip on first iteration
        if self.iteration < 1:
            return None

        if not self._upd_alpha:
            return None

        # current coord must have en, grad
        assert self._coords.g is not None
        assert np.isfinite(self.last_energy_change)

        coords_l, coords_k = self._history.final, self._history.penultimate
        step = coords_l.raw - coords_k.raw
        cart_step = coords_l.to("cart") - coords_k.to("cart")
        cart_step_size = np.linalg.norm(cart_step)

        pred_energy_change = np.dot(
            coords_k.g, step
        ) + 0.5 * np.linalg.multi_dot((step, coords_k.h, step))

        # ratio between actual deltaE and predicted deltaE
        ratio = self.last_energy_change / pred_energy_change

        if ratio < 0.25:
            self.alpha = max(self.alpha / 2, self._min_alpha)
        elif 0.25 < ratio < 0.75:
            pass
        elif 0.75 < ratio < 1.25:
            # if step taken is within 5% of the trust radius
            if abs(cart_step_size - self.alpha) / self.alpha < 0.05:
                self.alpha = min(self.alpha * 1.414, self._max_alpha)
        else:  # ratio > 1.25
            # also decrease if ratio too high
            self.alpha = max(self.alpha / 2, self._min_alpha)

        print(f"Ratio = {ratio}, Current trust radius = {self.alpha} Angstrom")

        return None
