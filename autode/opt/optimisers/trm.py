"""
A hybrid RFO (Rational Function Optimisation) and
Trust Radius Model (TRM) optimiser. Based upon the
optimisers available in multiple popular QM softwares.

Only minimiser, this is not TS search/constrained optimiser
"""
import numpy as np
from scipy.optimize import root_scalar
from itertools import combinations
from typing import TYPE_CHECKING, Tuple, Any, Union

from autode.opt.coordinates import CartesianCoordinates, DIC, OptCoordinates
from autode.opt.coordinates.primitives import PrimitiveDistance
from autode.opt.optimisers import CRFOptimiser
from autode.opt.optimisers.hessian_update import BFGSSR1Update
from autode.exceptions import OptimiserStepError
from autode.values import GradientRMS, PotentialEnergy, Distance
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class HybridTRMOptimiser(CRFOptimiser):
    """
    A hybrid of RFO and TRM/QA optimiser, with dynamic trust
    radius. If the RFO step is smaller in magnitude than trust radius,
    then the step-length is controlled by using the Trust-Radius Model
    (TRM), also sometimes known as Quadratic Approximation (QA).
    It falls-back on simple scalar scaling of RFO step, if the
    TRM/QA method fails to converge to the required step-size. The
    trust radius is updated by a modification of Fletcher's method.

    See References:

    [1] P. Culot et. al, Theor. Chim. Acta, 82, 1992, 189-205

    [2] T. Helgaker, Chem. Phys. Lett., 182(5), 1991, 503-510

    [3] J. T. Golab et al., Chem. Phys., 78, 1983, 175-199

    [4] R. Fletcher, Practical Methods of Optimization, Wiley, Chichester, 1981
    """

    def __init__(
        self,
        init_trust: Union[Distance, float] = 0.1,
        update_trust: bool = True,
        min_trust: Union[Distance, float] = 0.01,
        max_trust: Union[Distance, float] = 0.3,
        damp: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Hybrid RFO and TRM/QA optimiser with dynamic trust radius (and
        optional damping if oscillation in energy and gradient is detected)

        ---------------------------------------------------------------------
        Args:
            init_trust: Initial value of trust radius in Angstrom, this determines
                        the maximum norm of Cartesian step size
            update_trust: Whether to update the trust radius or not
            min_trust: Minimum bound for trust radius (only for trust update)
            max_trust: Maximum bound for trust radius (only for trust update)
            damp: Whether to apply damping if oscillation is detected
            *args: Additional arguments for ``NDOptimiser``
            **kwargs: Additional keyword arguments for ``NDOptimiser``

        Keyword Args:
            maxiter (int): Maximum number of iterations
            gtol (GradientRMS): Tolerance on RMS(|∇E|) in Cartesian coordinates
            etol (PotentialEnergy): Tolerance on |E_i+1 - E_i|

        See Also:
            :py:meth:`NDOptimiser <NDOptimiser.__init__>`
        """
        _ = kwargs.pop("init_alpha", None)
        super().__init__(init_alpha=init_trust, *args, **kwargs)  # type: ignore

        assert 0.0 < min_trust < max_trust
        self._max_alpha = Distance(max_trust, "ang")
        self._min_alpha = Distance(min_trust, "ang")
        self._upd_alpha = bool(update_trust)

        self._damping_on = bool(damp)
        self._last_damped_iteration = 0

        self._hessian_update_types = [BFGSSR1Update]

    @property
    def _g_norm(self) -> GradientRMS:
        """Calculate the RMS gradient norm in Cartesian coordinates"""
        assert self._coords is not None, "Must have coordinates"
        grad = self._coords.to("cart").g
        return GradientRMS(np.sqrt(np.average(np.square(grad))))

    @property
    def converged(self) -> bool:
        if self._species is not None and self._species.n_atoms == 1:
            return True  # Optimisation 0 DOF is always converged

        # gradient is better indicator of stationary point
        if self._g_norm <= self.gtol / 10:
            logger.warning(
                f"Gradient norm criteria overachieved. "
                f"{self._gtol.to('Ha/ang')/10:.3f} Ha/Å. "
                f"Signalling convergence"
            )
            return True

        # both gradient and energy must be converged!!
        return self._abs_delta_e < self.etol and self._g_norm < self.gtol

    def _initialise_species_and_method(
        self,
        species: "Species",
        method: "Method",
    ) -> None:
        """
        Initialise species and method while checking that
        there are no constraints in the species
        """
        super()._initialise_species_and_method(species, method)
        assert self._species is not None, "Must have a species now"
        assert (
            not self._species.constraints.any
        ), "HybridTRMOptimiser cannot work with constraints!"
        return None

    def _step(self) -> None:
        """
        Hybrid RFO/TRM step; if the RFO step is larger than the trust
        radius, it switches to TRM model to get the best step within
        the trust radius, if that calculation fails, it simply scales
        back the RFO step to lie within trust radius
        """
        assert self._coords is not None, "Must have coordinates"

        self._coords.h = self._updated_h()
        self._update_trust_radius()

        if self._damping_on and self._is_oscillating():
            self._damped_step()
            logger.info("Skipping quasi-NR step after damping")
            return None

        self._coords.allow_unconverged_back_transform = True

        rfo_h_eff = self._get_rfo_minimise_h_eff(self._coords)
        rfo_step, rfo_step_size = self._get_step_and_cart_size_from_h_eff(
            self._coords, rfo_h_eff
        )

        if rfo_step_size < self.alpha:
            logger.info("Taking a pure RFO step")
            step = rfo_step
        else:
            try:
                qa_h_eff = self._get_trm_minimise_h_eff(self._coords)
                qa_step, _ = self._get_step_and_cart_size_from_h_eff(
                    self._coords, qa_h_eff
                )
                logger.info("Taking a TRM/QA step optimised to trust radius")
                step = qa_step

            except OptimiserStepError as exc:
                logger.debug(f"TRM/QA step failed: {str(exc)}")
                # if TRM/QA failed, take scaled RFO step
                scaled_step = rfo_step * (self.alpha / rfo_step_size)

                logger.info(
                    "Taking a scaled RFO step approximately within "
                    "trust radius"
                )
                step = scaled_step

        self._coords.allow_unconverged_back_transform = False
        self._coords = self._coords + step

        step_size = np.linalg.norm(
            self._coords.to("cart") - self._history.penultimate.to("cart")  # type: ignore
        )
        logger.info(f"Size of step taken (in Cartesian) = {step_size:.3f} Å")
        return None

    @staticmethod
    def _get_step_and_cart_size_from_h_eff(
        old_coords: OptCoordinates, hess_eff: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Obtains the Cartesian step size given a step in the current
        coordinate system (e.g., Delocalised Internal Coordinates)

        Args:
            old_coords (OptCoordinates): previous coordinates
            hess_eff (np.ndarray): Effective (shifted) hessian

        Returns:
            (float): The step size in Cartesian
        """
        step = np.matmul(-np.linalg.inv(hess_eff), old_coords.g)
        new_coords = old_coords + step
        cart_delta = new_coords.to("cart") - old_coords.to("cart")
        return step, float(np.linalg.norm(cart_delta))

    @staticmethod
    def _get_rfo_minimise_h_eff(coords) -> np.ndarray:
        """
        Using current Hessian and gradient, obtain the level-shifted
        Hessian that would provide a minimising RFO step

        Returns:
            (np.ndarray): The level-shifted effective Hessian
        """
        h_n = coords.h.shape[0]

        # form the augmented Hessian
        aug_h = np.zeros(shape=(h_n + 1, h_n + 1))
        aug_h[:h_n, :h_n] = coords.h
        aug_h[-1, :h_n] = coords.g
        aug_h[:h_n, -1] = coords.g

        aug_h_lmda, aug_h_v = np.linalg.eigh(aug_h)

        # RFO step uses the lowest non-zero eigenvalue
        if abs(aug_h_lmda[0]) < 1.0e-14:
            raise OptimiserStepError(
                "First mode is 0, handling such cases are not implemented"
            )
        lmda = aug_h_lmda[0]

        # effective hessian = H - lambda * I
        return coords.h - lmda * np.eye(h_n)

    def _get_trm_minimise_h_eff(self, coords) -> np.ndarray:
        """
        Using current Hessian and gradient, get the level-shifted Hessian
        for a minimising step, whose magnitude (norm) is approximately
        equal to the trust radius (TRM or QA step) in Cartesian coordinates.

        Described in J. Golab, D. L. Yeager, and P. Jorgensen,
        Chem. Phys., 78, 1983, 175-199

        Args:
            coords (OptCoordinates): current coordinates

        Returns:
            (np.ndarray): The level-shifted Hessian for TRM/QA step
        """
        h_n = coords.h.shape[0]
        h_eigvals = np.linalg.eigvalsh(coords.h)
        if abs(h_eigvals[0]) < 1.0e-14:
            raise OptimiserStepError(
                "First mode is 0, handling such cases are not implemented"
            )
        first_b = h_eigvals[0]  # first non-zero eigenvalue of H

        def get_internal_step_size_and_deriv(lmda):
            """Get the internal coordinate step, step size and
            the derivative for the given lambda"""
            shifted_h = coords.h - lmda * np.eye(h_n)
            inv_shifted_h = np.linalg.inv(shifted_h)
            step = -inv_shifted_h @ coords.g
            size = np.linalg.norm(step)
            deriv = -np.linalg.multi_dot((step, inv_shifted_h, step))
            deriv = float(deriv) / size
            return size, deriv, step

        def optimise_lambda_for_int_step(int_size):
            """
            Given a step length in internal coords, get the
            value of lambda that will give that step.
            Here f(λ) represents the internal coords step size
            as a function of lambda.
            """
            # use bisection to ensure lambda < first_b
            d = 1.0
            found_upper_lim = False
            for _ in range(20):
                size, _, _ = get_internal_step_size_and_deriv(first_b - d)
                # find f(x) > 0, so going downhill has no risk of jumping over
                if size > int_size:
                    found_upper_lim = True
                    break
                d = d / 2.0
            if not found_upper_lim:
                raise OptimiserStepError("Failed to find f(λ) > 0")

            lmda_guess = first_b - d
            found_step_size = False
            for _ in range(50):
                size, der, _ = get_internal_step_size_and_deriv(lmda_guess)
                if abs(size - int_size) / int_size < 0.001:  # 0.1% error
                    found_step_size = True
                    break
                lmda_guess -= (1 - size / int_size) * (size / der)
            if not found_step_size:
                raise OptimiserStepError("Failed in optimising internal step")
            return lmda_guess

        last_lmda = 0.0  # initialize non-local var

        def cart_step_length_error(int_size):
            """
            Deviation from trust radius in Cartesian,
            given step-size in internal coordinates
            """
            nonlocal last_lmda
            last_lmda = optimise_lambda_for_int_step(int_size)
            _, _, step = get_internal_step_size_and_deriv(last_lmda)
            step_size = np.linalg.norm(
                (coords + step).to("cart") - coords.to("cart")
            )
            return step_size - self.alpha

        # The value of shift parameter lambda must lie within (-infinity, first_b)
        # Need to find the roots of the 1D function cart_step_length_error
        int_step_size = float(self.alpha)
        # starting guess size
        fac = 1.0
        size_min_bound = None
        size_max_bound = None
        found_bounds = False
        for _ in range(10):
            int_step_size = int_step_size * fac
            err = cart_step_length_error(int_step_size)

            if err < -1.0e-3:  # found where error is < 0
                fac = 2.0  # increase the step size
                size_min_bound = int_step_size

            elif err > 1.0e-3:  # found where error is > 0
                fac = 0.5  # decrease the step size
                size_max_bound = int_step_size

            else:  # found err ~ 0 already!, no need for root finding
                return coords.h - last_lmda * np.eye(h_n)

            if (size_max_bound is not None) and (size_min_bound is not None):
                found_bounds = True
                break

        if not found_bounds:
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

        if last_lmda > min(0, first_b):
            raise OptimiserStepError("Unknown error in finding optimal lambda")

        logger.debug(f"Optimised lambda for QA step: {last_lmda}")
        # use the final lambda to construct level-shifted Hessian
        return coords.h - last_lmda * np.eye(h_n)

    def _update_trust_radius(self) -> None:
        """
        Updates the trust radius before a geometry step
        """
        assert self._coords is not None, "Must have coordinates"

        # skip on first iteration
        if self.iteration < 1:
            return None

        if not self._upd_alpha:
            return None

        # current coord must have energy
        assert self._coords.e is not None
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
            self.alpha = max(Distance(set_trust), self._min_alpha)
        elif 0.25 < ratio < 0.75:
            pass
        elif 0.75 < ratio < 1.25:
            # if step taken is within 5% of the trust radius
            if abs(cart_step_size - self.alpha) / self.alpha < 0.05:
                self.alpha = min(Distance(1.414 * self.alpha), self._max_alpha)
        elif 1.25 < ratio < 1.5:
            pass
        else:  # ratio > 1.5
            set_trust = 0.5 * min(self.alpha, Distance(cart_step_size))
            self.alpha = max(set_trust, self._min_alpha)

        if (ratio < -1.0) or (ratio > 2.0):
            logger.warning(
                "Energy increased/decreased by large amount,"
                "rejecting last geometry step"
            )
            # remove the last geometry from history
            self._history.pop()

        logger.info(
            f"Ratio = {ratio:.3f}, Current trust radius = {self.alpha:.3f} Å"
        )
        return None

    def _is_oscillating(self) -> bool:
        """
        Check whether the optimiser is oscillating instead of converging.
        If both the energy is oscillating (i.e. up-> down or down->up) in
        the last two steps, and the energy has not gone below the lowest
        energy in the last 4 iterations, then it is assumed that the
        optimiser is oscillating

        Returns:
            (bool): True if oscillation is detected, False otherwise
        """
        # allow the optimiser 4 free iterations before checking oscillation
        if self.iteration - self._last_damped_iteration < 4:
            return False

        # energy change two steps before i.e. -3, -2
        e_change_before_last = self._history[-2].e - self._history[-3].e

        # sign of changes should be different if E oscillating
        if not self.last_energy_change * e_change_before_last < 0:
            return False

        # check if energy has gone down since the last 4 iters
        min_index = np.argmin([coord.e for coord in self._history])
        if min_index < (self.iteration - 4):
            logger.warning(
                "Oscillation detected in optimiser, energy has "
                "not decreased in 4 iterations"
            )
            return True

        return False

    def _damped_step(self) -> None:
        """
        Take a damped step by interpolating between the last two coordinates
        """
        assert self._coords is not None, "Must have coordinates"

        logger.info("Taking a damped step...")
        self._last_damped_iteration = self.iteration

        # is halfway interpolation good?
        new_coords_raw = (self._coords.raw + self._history[-2].raw) / 2
        damped_step = new_coords_raw - self._coords.raw
        self._coords = self._coords + damped_step
        return None


class CartesianHybridTRMOptimiser(HybridTRMOptimiser):
    def _initialise_run(self) -> None:
        """Initialise the optimisation with Cartesian coordinates"""
        assert self._species is not None, "Must have a species"
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._update_gradient_and_energy()
        return None
