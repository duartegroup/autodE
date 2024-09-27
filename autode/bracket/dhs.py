"""
Dewar-Healy-Stewart Method for finding transition states

Also implements DHS-GS, CI-DHS and CI-DHS-GS methods

[1] M. J. S. Dewar, E. Healy, J. Chem. Soc. Farady Trans. 2, 1984, 80, 227-233
"""
import numpy as np
from typing import Tuple, Union, Optional, Any, TYPE_CHECKING
from enum import Enum

from autode.values import Distance, Angle, GradientRMS, PotentialEnergy
from autode.bracket.imagepair import EuclideanImagePair
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.utils import TruncatedTaylor
from autode.opt.optimisers.hessian_update import BFGSSR1Update
from autode.bracket.base import BaseBracketMethod
from autode.opt.optimisers import RFOptimiser, ConvergenceParams
from autode.exceptions import OptimiserStepError
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method
    from autode.opt.optimisers.base import ConvergenceTolStr


class DistanceConstrainedOptimiser(RFOptimiser):
    """
    Constrained optimisation of a molecule, with the Euclidean
    distance being kept constrained to a fixed value. The
    constraint is enforced by a Lagrangian multiplier. An optional
    linear search can be done to speed up convergence.

    Same concept as that used in the corrector step of
    Gonzalez-Schlegel second-order IRC integrator. However,
    current implementation is modified to take steps within
    a trust radius.

    [1] C. Gonzalez, H. B. Schlegel, J. Chem. Phys., 90, 1989, 2154
    """

    def __init__(
        self,
        pivot_point: Optional[CartesianCoordinates],
        init_trust: float = 0.1,
        line_search: bool = True,
        angle_thresh: Angle = Angle(5, units="deg"),
        old_coords_read_hess: Optional[CartesianCoordinates] = None,
        *args,
        **kwargs,
    ):
        """
        Initialise a distance constrained optimiser. The pivot point
        is the point against which the distance is constrained. Optionally
        a linear search can be used to attempt to speed up convergence, but
        it may not improve performance in all cases.

        Args:
            init_trust: Initial trust radius in Angstrom
            pivot_point: Coordinates of the pivot point
            line_search: Whether to use linear search
            angle_thresh: An angle threshold above which linear search
                        will be rejected (in Degrees)
            old_coords_read_hess: Old coordinate with hessian which will
                        be used to obtain the initial hessian by a
                        Hessian update scheme
        """
        kwargs.pop("init_alpha", None)
        super().__init__(*args, init_alpha=init_trust, **kwargs)

        if not isinstance(pivot_point, CartesianCoordinates):
            raise NotImplementedError(
                "Internal coordinates are not implemented in distance"
                "constrained optimiser right now, please use Cartesian"
            )
        self._pivot = pivot_point
        self._do_line_search = bool(line_search)
        self._angle_thresh = Angle(angle_thresh, units="deg").to("radian")
        self._target_dist: Optional[float] = None

        self._hessian_update_types = [BFGSSR1Update]
        self._old_coords = old_coords_read_hess

    def _initialise_run(self) -> None:
        """Initialise self._coords, gradient and hessian"""
        assert self._species is not None, "Must have a species to init run"

        self._coords = CartesianCoordinates(self._species.coordinates)
        self._target_dist = np.linalg.norm(self.dist_vec)
        self._update_gradient_and_energy()

        # Update the Hessian from old coordinates, if exists
        if self._old_coords is not None and self._old_coords.h is not None:
            assert isinstance(self._old_coords, CartesianCoordinates)
            self._coords.update_h_from_old_h(
                self._old_coords, self._hessian_update_types
            )
        else:
            # no hessian available, use low level method
            self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
            self._coords.make_hessian_positive_definite()

    @property
    def converged(self) -> bool:
        """Has the optimisation converged"""
        assert self._coords is not None

        # Check only the tangential component of gradient
        g_tau = self.tangent_grad
        rms_g_tau = np.sqrt(np.mean(np.square(g_tau)))
        max_g_tau = np.max(np.abs(g_tau))

        curr_params = self._history.conv_params()
        curr_params.rms_g = GradientRMS(rms_g_tau, "Ha/ang")
        curr_params.max_g = GradientRMS(max_g_tau, "Ha/ang")
        return self.conv_tol.meets_criteria(curr_params)

    @property
    def tangent_grad(self) -> np.ndarray:
        """
        Obtain the component of atomic gradients tangent to the distance
        vector between current coords and pivot point
        """
        assert self._coords is not None and self._coords.g is not None

        grad = self._coords.g
        # unit vector in the direction of distance vector
        d_hat = self.dist_vec / np.linalg.norm(self.dist_vec)
        tangent_grad = grad - (grad.dot(d_hat)) * d_hat
        return tangent_grad

    @property
    def dist_vec(self) -> np.ndarray:
        """
        Get the distance vector (p) between the current coordinates
        and the pivot point

        Returns:
            (np.ndarray):
        """
        return np.array(self._coords - self._pivot)

    def _update_gradient_and_energy(self) -> None:
        # Hessian update is done after en grad calculation, not in step
        # so that it is present in the final converged coords, which
        # can be used to start off the next batch of optimisation
        super()._update_gradient_and_energy()
        if self.iteration != 0:
            assert self._coords is not None, "Must have set coordinates"
            self._coords.update_h_from_old_h(
                self._history.penultimate, self._hessian_update_types
            )

    def _step(self) -> None:
        """
        A step that maintains the distance of the coordinate from
        the pivot point. A line search is done if it is not the first
        iteration (and it has not been turned off), and then a
        quasi-Newton step with a Lagrangian constraint for the distance
        is taken (falls back to steepest descent with projected gradient
        if this fails).
        """
        assert self._coords is not None, "Must have set coordinates"

        # if energy is rising, interpolate halfway between last step
        if self.iteration >= 1 and (
            self.last_energy_change > PotentialEnergy(5, "kcalmol")
        ):
            logger.warning("Energy rising, going back half a step")
            half_interp = (self._coords + self._history.penultimate) / 2
            self._coords = half_interp
            return None

        if self.iteration >= 1 and self._do_line_search:
            coords, grad = self._line_search_on_sphere()
        else:
            coords, grad = self._coords, self._coords.g

        try:
            step = self._get_lagrangian_step(coords, grad)
            logger.info(
                f"Taking a quasi-Newton step: {np.linalg.norm(step):.3f} Å"
            )
        except OptimiserStepError:
            step = self._get_sd_step(coords, grad)
            logger.warning(
                f"Failed to take quasi-Newton step, taking steepest "
                f"descent step instead: {np.linalg.norm(step):.3f} Å"
            )

        # the step is on the interpolated coordinates (if done)
        actual_step = (coords + step) - self._coords
        self._coords = self._coords + actual_step
        return None

    def _get_sd_step(self, coords, grad) -> np.ndarray:
        """
        Obtain a steepest descent step minimising the tangential
        gradient. This step cannot perfectly maintain the same
        distance from pivot point. The step size is at most half
        of the trust radius.

        Args:
            coords: Previous coordinates
            grad: Previous gradient

        Returns:
            (np.ndarray): Step in Cartesian coordinates
        """
        dist_vec = coords - self._pivot
        dist_hat = dist_vec / np.linalg.norm(dist_vec)
        perp_grad = grad - np.dot(grad, dist_hat) * dist_hat

        sd_step = -perp_grad
        if np.linalg.norm(sd_step) > self.alpha / 2:
            sd_step *= (self.alpha / 2) / np.linalg.norm(sd_step)

        return sd_step

    def _get_lagrangian_step(self, coords, grad) -> np.ndarray:
        """
        Obtain the step that will minimise the gradient tangent to
        the distance vector from pivot point, while maintaining the
        same distance from pivot point. Takes the step within current
        trust radius.

        Args:
            coords: Previous coordinate (either from quasi-NR step
                    or from linear search)
            grad: Previous gradient (either from quasi-NR or linear
                  search)

        Returns:
            (np.ndarray): Step in cartesian (or mw-cartesian) coordinates

        Raises:
            OptimiserStepError: If scipy fails to calculate constrained step
        """
        from scipy.optimize import minimize

        assert self._coords is not None, "Must have set coordinates"

        # NOTE: Since the linear interpolation should produce a point
        # in the vicinity of the last two points, it seems reasonable to
        # also use the hessian from the last point in the case of linear
        # interpolation being done
        taylor_pes = TruncatedTaylor(coords, grad, self._coords.h)

        def step_size_constr(x):
            """step size must be <= trust radius"""
            step_est = x - coords
            # inequality constraint, should be > 0
            return self.alpha - np.linalg.norm(step_est)

        def lagrangian_constr(x):
            """step must maintain same distance from pivot"""
            p = x - self._pivot
            return np.linalg.norm(p) - self._target_dist

        constrs = (
            {"type": "ineq", "fun": step_size_constr},
            {"type": "eq", "fun": lagrangian_constr},
        )
        # NOTE: The Lagrangian constraint should be ideally calculated using
        # a multiplier which can be found by a 1-D root search, however, it
        # seems to produce really large steps. So instead the constraint
        # and the trust radius are both enforced by doing a constrained
        # optimisation on the truncated Taylor surface, which should give
        # a quadratic step that follows the constraint and is within trust
        # radius

        res = minimize(
            fun=taylor_pes.value,
            x0=np.array(self._coords),
            method="slsqp",
            jac=taylor_pes.gradient,
            options={"maxiter": 2000},
            constraints=constrs,
        )

        if not res.success:
            raise OptimiserStepError(
                f"Unable to obtain distance-constrained step\nResult: {res}"
            )

        step = res.x - coords
        return step

    def _line_search_on_sphere(
        self,
    ) -> Tuple[Optional[CartesianCoordinates], Optional[np.ndarray]]:
        """
        Linear search on a hypersphere of radius equal to the target
        distance.

        Returns:
            (Tuple): Interpolated coordinates and gradient as tuple
        """
        assert self._coords is not None, "Must have set coords to line search"

        # Eq (12) to (15) in J. Chem. Phys., 90, 1989, 2154
        # Notation follows the publication
        last_coords = self._history[-2]
        assert last_coords is not None

        p_prime = self.dist_vec
        g_prime_per = self._coords.g - p_prime * (
            np.dot(self._coords.g, p_prime) / np.dot(p_prime, p_prime)
        )
        g_prime_per = np.linalg.norm(g_prime_per)
        p_prime_prime = np.array(last_coords - self._pivot)
        g_prime_prime_per = last_coords.g - p_prime_prime * (
            np.dot(last_coords.g, p_prime_prime)
            / np.dot(p_prime_prime, p_prime_prime)
        )
        g_prime_prime_per = np.linalg.norm(g_prime_prime_per)
        cos_theta_prime = np.dot(p_prime, p_prime_prime) / (
            np.linalg.norm(p_prime) * np.linalg.norm(p_prime_prime)
        )
        assert -1 < cos_theta_prime < 1
        theta_prime = np.arccos(cos_theta_prime)
        theta = (g_prime_prime_per * theta_prime) / (
            g_prime_prime_per - g_prime_per
        )

        p_interp = p_prime_prime * (
            np.cos(theta)
            - np.sin(theta) * np.cos(theta_prime) / np.sin(theta_prime)
        )
        p_interp += p_prime * (np.sin(theta) / np.sin(theta_prime))

        g_interp = last_coords.g * (1 - theta / theta_prime)
        g_interp += self._coords.g * (theta / theta_prime)

        x_interp = self._pivot + p_interp

        step_size = np.linalg.norm(x_interp - self._coords)
        angle_change = abs(theta_prime - theta)
        if (
            (
                angle_change > self._angle_thresh
                and abs(theta) > self._angle_thresh
            )
            or (
                theta < 0  # extrapolating instead of interpolating
                and theta_prime < self._angle_thresh
            )
            or (step_size > self.alpha)  # larger than trust radius
        ):
            logger.info("Linear interpolation step is unstable, skipping")
            return self._coords, self._coords.g

        logger.info(f"Linear interpolation - step size: {step_size:.3f} Å")

        return x_interp, g_interp


class ImageSide(Enum):
    """Represents one side of the image-pair"""

    left = 0
    right = 1


class DHSImagePair(EuclideanImagePair):
    """
    Image-pair used for Dewar-Healy-Stewart (DHS) method to
    find transition states. In this method, only one side is
    modified in a step, so functions to work with only one
    side is present here
    """

    @property
    def ts_guess(self) -> Optional["Species"]:
        """
        In DHS method, the images can only rise in energy; therefore,
        the highest energy image is the ts_guess. If CI-NEB is run,
        then that result is returned instead
        """
        tmp_spc = self._left_image.new_species(name="peak")

        if self._cineb_coords is not None:
            assert (
                self._cineb_coords is not None
                and self._cineb_coords.e
                and self._cineb_coords.g is not None
            )
            tmp_spc.coordinates = self._cineb_coords
            tmp_spc.energy = self._cineb_coords.e
            tmp_spc.gradient = self._cineb_coords.g.reshape(-1, 3).copy()
            return tmp_spc

        # NOTE: Even though the final points are probably the highest
        # this is not guaranteed, due to the probability of one end
        # jumping over the barrier. So we iterate through all coords

        energies = []
        max_e = PotentialEnergy(-np.inf)
        peak_coords: Optional[CartesianCoordinates] = None
        for coord in self._total_history:
            energies.append(coord.e)
            if coord.e is None:
                logger.error(
                    "Energy values are missing in the trajectory of this"
                    " image-pair. Unable to obtain transition state guess"
                )
                return None
            if coord.e > max_e:
                max_e = coord.e
                peak_coords = coord

        assert peak_coords is not None
        tmp_spc.coordinates = peak_coords
        tmp_spc.energy = peak_coords.e
        if peak_coords.g is not None:
            tmp_spc.gradient = peak_coords.g.reshape(-1, 3).copy()
        return tmp_spc

    def get_coord_by_side(self, side: ImageSide) -> CartesianCoordinates:
        """For external usage, supplies the coordinate object by side"""
        if side == ImageSide.left:
            return self.left_coords
        elif side == ImageSide.right:
            return self.right_coords
        else:
            raise ValueError

    def put_coord_by_side(
        self, new_coord: CartesianCoordinates, side: ImageSide
    ) -> None:
        """
        For external usage, puts the new coordinate in appropriate side

        Args:
            new_coord (CartesianCoordinates): New set of coordinates
            side (ImageSide): left or right
        """
        if side == ImageSide.left:
            self.left_coords = new_coord
        elif side == ImageSide.right:
            self.right_coords = new_coord
        else:
            raise ValueError
        return None

    def get_last_step_by_side(self, side: ImageSide) -> Optional[np.ndarray]:
        """
        Obtain the last step on the provided side (for the Growing
        String like step)
        """
        if side == ImageSide.left:
            hist = self._left_history
        elif side == ImageSide.right:
            hist = self._right_history
        else:
            raise ValueError

        if len(hist) < 2:
            return None
        return hist.final - hist.penultimate

    def get_dhs_step_by_side(
        self, side: ImageSide, step_size: float
    ) -> np.ndarray:
        """
        Obtain the DHS extrapolation step on the specified side,
        with the specified step size

        Args:
            side (ImageSide): left or right
            step_size (float): Step size in Angstrom

        Returns:
            (np.ndarray): The step
        """
        dhs_step = self.dist_vec * (step_size / self.dist)
        if side == ImageSide.left:
            dhs_step *= -1.0
        elif side == ImageSide.right:
            pass
        else:
            raise ValueError

        return dhs_step


class DHS(BaseBracketMethod):
    """
    Dewar-Healy-Stewart method for finding transition states,
    from the reactant and product structures
    """

    def __init__(
        self,
        initial_species: "Species",
        final_species: "Species",
        large_step: Union[Distance, float] = Distance(0.2, "ang"),
        small_step: Union[Distance, float] = Distance(0.05, "ang"),
        switch_thresh: Union[Distance, float] = Distance(1.5, "ang"),
        conv_tol: Union["ConvergenceParams", "ConvergenceTolStr"] = "loose",
        **kwargs,
    ):
        """
        Dewar-Healy-Stewart method to find transition states. The distance
        tolerance convergence criteria should not be much lower than 0.5 Angstrom
        as DHS is unstable when the distance is low, and there is a tendency for
        one image to jumpy over the barrier.

        Args:
            initial_species: The "reactant" species

            final_species: The "product" species

            large_step: The size of the DHS step when distance between the
                        images is larger than switch_thresh (Angstrom)

            small_step: The size of the DHS step when distance between the
                        images is smaller than swtich_thresh (Angstrom)

            switch_thresh: When distance between the two images is less than
                        this cutoff, smaller DHS extrapolation steps are taken

            conv_tol: Convergence tolerance for the distance-constrained
                      optimiser

        Keyword Args:

            maxiter: Maximum number of en/grad evaluations

            dist_tol: The distance tolerance at which DHS will
                      stop, values less than 0.5 Angstrom are not
                      recommended.

            cineb_at_conv: Whether to run CI-NEB calculation from the end
                           points after the DHS is converged
        """
        super().__init__(initial_species, final_species, **kwargs)

        # imgpair is only used for storing the points here
        self.imgpair: DHSImagePair = DHSImagePair(
            initial_species, final_species
        )

        # DHS needs to keep an extra reference method and n_cores
        self._method: Optional[Method] = None
        self._n_cores: Optional[int] = None

        self._large_step = Distance(abs(large_step), "ang")
        self._small_step = Distance(abs(small_step), "ang")
        self._sw_thresh = Distance(abs(switch_thresh), "ang")
        assert self._small_step < self._large_step
        self._conv_tol = conv_tol

        self._step_size: Optional[Distance] = None
        if self._large_step > self.imgpair.dist:
            logger.warning(
                f"Step size ({self._large_step:.3f} Å) for {self._name}"
                f" is larger than the starting Euclidean distance between"
                f" images ({self.imgpair.dist:.3f} Å). This calculation"
                f" will likely run into errors."
            )

        # NOTE: In DHS the micro-iterations are done separately in
        # an optimiser, so keep track with local variable
        self._current_microiters: int = 0

    def _initialise_run(self) -> None:
        """
        Initialise energies/gradients for the first DHS macro-iteration
        """
        self.imgpair.update_both_img_engrad()
        return None

    def _step(self) -> None:
        """
        A DHS step consists of a macro-iteration step, where a step along
        the linear path between two images is taken, and several micro-iteration
        steps in the distance-constrained optimiser, to return to the MEP
        """
        assert self._method is not None, "Must have a set method"
        assert self.imgpair.left_coords.e and self.imgpair.right_coords.e

        if self.imgpair.dist > self._sw_thresh:
            self._step_size = self._large_step
        else:
            self._step_size = self._small_step
        opt_trust = min(self._step_size, Distance(0.1, "ang"))

        if self.imgpair.left_coords.e < self.imgpair.right_coords.e:
            side = ImageSide.left
            pivot = self.imgpair.right_coords
        else:
            side = ImageSide.right
            pivot = self.imgpair.left_coords

        old_coords: Any = self.imgpair.get_coord_by_side(side)
        old_coords = old_coords if old_coords.h is not None else None
        # take a DHS step on the side with lower energy
        new_coord = self._get_dhs_step(side)

        # calculate the number of remaining maxiter to feed into optimiser
        curr_maxiter = self._maxiter - self._current_microiters
        if curr_maxiter <= 0:
            return None

        opt = DistanceConstrainedOptimiser(
            maxiter=curr_maxiter,
            conv_tol=self._conv_tol,
            init_trust=opt_trust,
            pivot_point=pivot,
            old_coords_read_hess=old_coords,
        )
        tmp_spc = self._species.copy()
        tmp_spc.coordinates = new_coord
        opt.run(tmp_spc, self._method, self._n_cores)
        self._micro_iter = self._micro_iter + opt.iteration

        # not converged can only happen if exceeded maxiter of optimiser
        if not opt.converged:
            return None

        rms_g_tau = np.sqrt(np.mean(np.square(opt.tangent_grad)))
        logger.info(
            "Successful optimization after DHS step, final RMS of "
            f"tangential gradient = {rms_g_tau:.6f} "
            f"Ha/angstrom"
        )

        # put results back into imagepair
        self.imgpair.put_coord_by_side(opt.final_coordinates, side)  # type: ignore
        opt.clean_up()
        return None

    def _calculate(
        self, method: "Method", n_cores: Optional[int] = None
    ) -> None:
        """
        Run the DHS calculation and CI-NEB if requested.

        Args:
            method (Method): Method used for calculating energy/gradients
            n_cores (int): Number of cores to use for calculation
        """
        self._method = method
        self._n_cores = n_cores
        super()._calculate(method, n_cores)

    @property
    def _macro_iter(self):
        """Total number of DHS steps taken so far"""
        # ImagePair only stores the converged coordinates, which
        # is equal to the number of macro-iterations (DHS steps)
        return self.imgpair.total_iters

    @property
    def _micro_iter(self) -> int:
        """Total number of optimiser steps in DHS"""
        return self._current_microiters

    @_micro_iter.setter
    def _micro_iter(self, value: int):
        """
        For DHS the number of microiters has to be manually
        set

        Args:
            value (int):
        """
        self._current_microiters = int(value)

    def _get_dhs_step(self, side: ImageSide) -> CartesianCoordinates:
        """
        Take a DHS step, on the side requested, along the distance
        vector between the two images, and return the new coordinates
        after taking the step

        Args:
            side (ImageSide): left or right

        Returns:
            (CartesianCoordinates): New predicted coordinates for that side
        """
        assert self._step_size is not None
        # take a DHS step of the size given
        dhs_step = self.imgpair.get_dhs_step_by_side(side, self._step_size)

        old_coord = self.imgpair.get_coord_by_side(side)
        new_coord = old_coord + dhs_step

        logger.info(
            f"DHS step on {side} image: taking a step of"
            f" size {np.linalg.norm(dhs_step):.4f} Å"
        )
        return new_coord


class DHSGS(DHS):
    """
    Dewar-Healy-Stewart method, augmented with Growing String (GS)
    method. The DHS step (stepping along the linear interpolated
    path between the two images) is mixed with a GS step (linear
    interpolation along last and current position of one image)
    in a fixed ratio.

    Proposed by J. Kilmes, D. R. Bowler, A. Michaelides,
    J. Phys.: Condens. Matter, 2010, 22(7), 074203
    """

    def __init__(self, *args, gs_mix: float = 0.5, **kwargs):
        """
        Arguments and other keyword arguments follow DHS, please
        see :py:meth:`DHS <autode.bracket.dhs.DHS.__init__>`

        Keyword Args:
            gs_mix (float): Represents the percentage of mixing of the
                            Growing String step with the DHS step. 0.3
                            means 0.3 * GS_step + (1-0.3) * DHS_step
                            It is not recommended to set this higher
                            than 0.5
        """
        super().__init__(*args, **kwargs)

        self._gs_mix = float(gs_mix)
        assert 0.0 < self._gs_mix < 1.0, "Mixing factor must be 0 < fac < 1"

    def _get_dhs_step(self, side: ImageSide) -> CartesianCoordinates:
        """
        Take a mixed DHS and GS step (interpolates between the two
        vectors) in the given ratio, and then return the new
        coordinates after taking the step

        Args:
            side (ImageSide):

        Returns:
            (CartesianCoordinates): New predicted coordinates for that side
        """
        assert self.imgpair is not None, "Must have an image pair"
        assert self._step_size is not None

        dhs_step = self.imgpair.get_dhs_step_by_side(side, self._step_size)
        gs_step = self.imgpair.get_last_step_by_side(side)

        if gs_step is None:
            gs_step = np.zeros_like(dhs_step)
            # hack to ensure the first step is 100% DHS (as GS is not possible)
            dhs_step = dhs_step / (1 - self._gs_mix)
        else:
            # rescale GS step as well so that one vector doesn't dominate
            gs_step *= np.linalg.norm(dhs_step) / np.linalg.norm(gs_step)

        old_coord = self.imgpair.get_coord_by_side(side)
        new_coord = (
            old_coord + (1 - self._gs_mix) * dhs_step + self._gs_mix * gs_step
        )
        # step size is variable due to adding GS component
        step_size = np.linalg.norm(new_coord - old_coord)
        logger.info(
            f"DHS-GS step on {side} image: taking a step "
            f"of size {step_size:.4f}"
        )

        return new_coord
