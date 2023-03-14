"""
Robust geometry optimisers, uses features from
multiple optimisation methods. (Similar to those
implemented in common QM packages)

Only minimisers, these are not TS search/constrained optimiser
"""
import numpy as np

from autode.opt.coordinates import CartesianCoordinates, DIC
from autode.opt.coordinates.primitives import Distance
from autode.opt.optimisers import CRFOptimiser, NDOptimiser
from autode.opt.optimisers.hessian_update import FlowchartUpdate
from autode.exceptions import OptimiserStepError
from autode.values import GradientRMS, PotentialEnergy
from autode.log import logger
from itertools import combinations


class HybridTRIMOptimiser(CRFOptimiser):
    """
    A hybrid of RFO and TRIM/QA optimiser, with dynamic trust
    radius. If the RFO step is smaller in magnitude than trust radius,
    then the step-length is controlled by using the Trust-Radius Image
    Minimisation method (TRIM), also known as Quadratic Approximation
    (QA). It falls-back on simple scalar scaling of RFO step, if the
    TRIM/QA method fails to converge to the required step-size. The
    trust radius is updated by a modification of Fletcher's method.

    See References:

    [1] P. Culot et. al, Theor. Chim. Acta, 82, 1992, 189-205

    [2] T. Helgaker, Chem. Phys. Lett., 182(5), 1991, 503-510

    [3] J. T. Golab et al., Chem. Phys., 78, 1983, 175-199
    """

    def __init__(
        self,
        coord_type: str = "dic",
        init_trust: float = 0.1,
        update_trust: bool = True,
        min_trust: float = 0.01,
        max_trust: float = 0.3,
        damp: bool = True,
        *args,
        **kwargs,
    ):
        """
        Hybrid RFO and TRIM/QA optimiser with dynamic trust radius (and
        optional damping if oscillation in energy and gradient is detected)

        ---------------------------------------------------------------------
        Args:
            coord_type: 'DIC' for delocalised internal coordinates, 'cart'
                               for Cartesian coordinates
            init_trust: Initial value of trust radius in Angstrom
            update_trust: Whether to update the trust radius or not
            min_trust: Minimum bound for trust radius (only for trust update)
            max_trust: Maximum bound for trust radius (only for trust update)
            damp: Whether to apply damping if oscillation is detected
            *args: Additional arguments for ``NDOptimiser``
            **kwargs: Additional keyword arguments for ``NDOptimiser``

        Keyword Args:
            maxiter (int): Maximum number of iterations (from ``NDOptimiser``)
            gtol (GradientRMS): Tolerance on RMS(|∇E|) (from ``NDOptimiser``)
            etol (PotentialEnergy): Tolerance on |E_i+1 - E_i|
                                    (from ``NDOptimiser``)

        See Also:
            :py:meth:`NDOptimiser <NDOptimiser.__init__>`
        """
        super().__init__(*args, init_alpha=init_trust, **kwargs)

        assert 0.0 < min_trust < max_trust
        self._max_alpha = float(max_trust)
        self._min_alpha = float(min_trust)
        self._upd_alpha = bool(update_trust)

        if coord_type.lower() in ["cart", "cartesian"]:
            self._coord_type = "cart"
        elif coord_type.lower() == "dic":
            self._coord_type = "dic"
        else:
            raise ValueError("Coordinate type must be either 'cart' or 'DIC'")
        # todo stop if any constraints detected -> is this needed
        self._should_damp = bool(damp)
        self._last_damp_iter = 0

        self._hessian_update_types = [FlowchartUpdate]

    @property
    def _g_norm(self) -> GradientRMS:
        """Calculate the RMS gradient norm in Cartesian coordinates"""
        grad = self._coords.to("cart").g
        return GradientRMS(np.sqrt(np.average(np.square(grad))))

    @property
    def converged(self) -> bool:
        if self._species is not None and self._species.n_atoms == 1:
            return True  # Optimisation 0 DOF is always converged

        # gradient is better indicator of stationary point
        if self._g_norm < self.gtol / 10:
            logger.warning(
                f"Gradient norm criteria overachieved. "
                f"{self._gtol.to('Ha/ang')/10:.3f} Ha/Å. "
                f"Signalling convergence"
            )

        # both gradient and energy must be converged!!
        return self._abs_delta_e < self.etol and self._g_norm < self.gtol

    def _initialise_run(self) -> None:
        self._build_coordinates()
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()

    def _build_coordinates(self) -> None:
        """Build delocalised internal coordinates"""
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

    def _step(self) -> None:

        self._coords.h = self._updated_h()
        self._update_trust_radius()

        # damping sets coords so return early
        if self._damp_if_required():
            logger.info("Skipping quasi-NR step after damping")
            return None

        self._coords.allow_unconverged_back_transform = True

        rfo_h_eff = self._get_rfo_minimise_h_eff()
        rfo_step = -np.linalg.inv(rfo_h_eff) @ self._coords.g
        rfo_step_size = self._get_cart_step_size_from_step(rfo_step)

        if rfo_step_size < self.alpha:
            logger.info("Taking a pure RFO step")
            step = rfo_step
        else:
            try:
                qa_h_eff = self._get_trim_minimise_h_eff()
                qa_step = -np.linalg.inv(qa_h_eff) @ self._coords.g
                logger.info("Taking a QA step optimised to trust radius")
                step = qa_step

            except OptimiserStepError as exc:
                logger.debug(f"QA step failed: {str(exc)}")
                # if QA failed, take scaled RFO step
                scaled_step = rfo_step * (self.alpha / rfo_step_size)

                logger.info(
                    "Taking a scaled RFO step approximately within "
                    "trust radius"
                )
                step = scaled_step

        step_size = self._get_cart_step_size_from_step(step)
        self._coords.allow_unconverged_back_transform = False
        self._coords = self._coords + step  # finally, take the step!
        logger.info(f"Size of step taken (in Cartesian) = {step_size:.3f} Å")

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

    def _get_trim_minimise_h_eff(self) -> np.ndarray:
        """
        Using current Hessian and gradient, get the level-shifted Hessian
        for a minimising step, whose magnitude (norm) is approximately
        equal to the trust radius (TRIM or QA step).

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
            """Get the internal coordinate step, step size and
            the derivative for the given lambda"""
            shifted_h = self._coords.h - lmda * np.eye(h_n)
            inv_shifted_h = np.linalg.inv(shifted_h)
            step = -inv_shifted_h @ self._coords.g
            size = np.linalg.norm(step)
            deriv = -np.linalg.multi_dot((step, inv_shifted_h, step))
            deriv = float(deriv) / size
            return size, deriv, step

        def optimise_lambda_for_int_step(int_size, lmda_guess):
            """Given a step length in internal coords, get the
            value of lambda that will give that step"""
            # from pyberny and geomeTRIC
            d = 1.0  # use bisection to ensure lambda < first_b
            for _ in range(20):
                size, _, _ = get_int_step_size_and_deriv(first_b - d)
                # find f(x) > 0, so going downhill has no risk of jumping over
                if size > int_size:
                    break
                d = d / 2.0
            else:
                raise OptimiserStepError("Failed to find f(x) > 0")
            lmda_guess = first_b - d
            for _ in range(50):
                size, der, _ = get_int_step_size_and_deriv(lmda_guess)
                if abs(size - int_size) / int_size < 0.001:  # 0.1% error
                    break
                lmda_guess -= (1 - size / int_size) * (size / der)
            else:
                raise OptimiserStepError("Failed in optimising internal step")
            return lmda_guess

        last_lmda = first_b - 1.0  # starting guess

        def cart_step_length_error(int_size):
            """Deviation for trust radius given step-size
            in internal coordinates"""
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

            if err < -1.0e-6:  # found where error is < 0
                fac = 2.0  # increase the step size
                size_min_bound = int_step_size

            elif err > 1.0e-6:  # found where error is > 0
                fac = 0.5  # decrease the step size
                size_max_bound = int_step_size

            else:  # found err ~ 0 already!, no need for root finding
                return self._coords.h - last_lmda * np.eye(h_n)

            if (size_max_bound is not None) and (size_min_bound is not None):
                break
        else:
            raise OptimiserStepError(
                "Unable to find bracket range for root finding"
            )
        # NOTE: The secant method does not need bracketing, however
        # the bracketing is done to ensure better convergence, as
        # the guesses for secant method should be closer to root

        logger.debug("Using secant method to find root")
        # Use scipy's root finder with secant method
        res = root_scalar(
            cart_step_length_error,
            method="secant",
            x0=size_min_bound,
            x1=size_max_bound,
            maxiter=20,
            rtol=0.01,  # 1% margin of error
        )

        if not res.converged:
            raise OptimiserStepError("Unable to converge step to trust radius")

        if not last_lmda < first_b:
            raise OptimiserStepError("Unknown error in finding optimal lambda")

        logger.debug(f"Optimised lambda for QA step: {last_lmda}")
        # use the final lambda to construct level-shifted Hessian
        return self._coords.h - last_lmda * np.eye(h_n)

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

        pred_e_change = np.dot(coords_k.g, step)
        pred_e_change += 0.5 * np.linalg.multi_dot((step, coords_k.h, step))

        # ratio between actual deltaE and predicted deltaE
        ratio = self.last_energy_change / float(pred_e_change)

        if ratio < 0.25:
            set_trust = 0.5 * min(self.alpha, cart_step_size)
            self.alpha = max(set_trust, self._min_alpha)
        elif 0.25 < ratio < 0.75:
            pass
        elif 0.75 < ratio < 1.25:
            # if step taken is within 5% of the trust radius
            if abs(cart_step_size - self.alpha) / self.alpha < 0.05:
                self.alpha = min(self.alpha * 1.414, self._max_alpha)
        elif 1.25 < ratio < 1.5:
            pass
        else:  # ratio > 1.5
            # if ratio is too high, scale down trust by a small amount
            self.alpha = max(self.alpha * 2 / 3, self._min_alpha)

        logger.info(f"Current trust radius = {self.alpha:.3f} Å")

        return None

    def _damp_if_required(self) -> bool:
        """
        If the energy and gradient norm are oscillating in the last three
        iterations, then interpolate between the last two coordinates to
        damp the oscillation (must skip the quasi-NR step afterwards)

        Returns:
            (bool): True if damped, False otherwise
        """
        # if user does not want, no damping
        if not self._should_damp:
            return False

        # allow the optimiser 3 free iterations before damping again
        if self.iteration - self._last_damp_iter < 3:
            return False

        # get last three coordinates
        coords_0, coords_1, coords_2, coords_3 = self._history[-4:]

        # energy changes in last three iters
        e_change_0_1 = coords_1.e - coords_0.e
        e_change_1_2 = coords_2.e - coords_1.e
        e_change_2_3 = coords_3.e - coords_2.e

        is_e_oscillating = (
            e_change_0_1 * e_change_1_2 < 0.0
        ) and (  # different sign
            e_change_1_2 * e_change_2_3 < 0.0
        )  # different sign

        # the sign of gradient norm change must also flip
        g_change_0_1 = np.linalg.norm(coords_1.to("cart").g) - np.linalg.norm(
            coords_0.to("cart").g
        )
        g_change_1_2 = np.linalg.norm(coords_2.to("cart").g) - np.linalg.norm(
            coords_1.to("cart").g
        )
        g_change_2_3 = np.linalg.norm(coords_3.to("cart").g) - np.linalg.norm(
            coords_2.to("cart").g
        )

        is_g_oscillating = (
            g_change_0_1 * g_change_1_2 < 0.0
        ) and (  # different sign
            g_change_1_2 * g_change_2_3 < 0.0
        )

        if is_e_oscillating and is_g_oscillating:
            logger.info("Oscillation detected in optimiser, damping")

            # is halfway interpolation good?
            new_coords_raw = (coords_2.raw + coords_3.raw) / 2.0
            step = new_coords_raw - self._coords.raw
            self._coords = self._coords + step

            self._last_damp_iter = self.iteration
            return True

        return False
