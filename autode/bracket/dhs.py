"""
Dewar-Healy-Stewart Method for finding transition states

As described in J. Chem. Soc. Farady Trans. 2, 1984, 80, 227-233
"""

from typing import Tuple, Union, Optional, TYPE_CHECKING
import numpy as np

from autode.values import Distance, Angle, GradientRMS
from autode.bracket.imagepair import ImagePair, ImgPairSideError
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate, BFGSUpdate
from autode.bracket.base import BaseBracketMethod
from autode.opt.optimisers import RFOptimiser

from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class _TruncatedTaylor:
    """The truncated taylor surface from current grad and hessian"""

    def __init__(
        self,
        centre: Union[OptCoordinates, np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
    ):
        self.centre = centre
        if hasattr(centre, "e") and centre.e is not None:
            self.en = centre.e
        else:
            # the energy can be relative and need not be absolute
            self.en = 0.0
        self.grad = grad
        self.hess = hess
        n_atoms = grad.shape[0]
        assert hess.shape == (n_atoms, n_atoms)

    def value(self, coords: np.ndarray):
        # E = E(0) + g^T . dx + 0.5 * dx^T. H. dx
        dx = coords - self.centre
        new_e = self.en + np.dot(self.grad, dx)
        new_e += 0.5 * np.linalg.multi_dot((dx, self.hess, dx))
        return new_e

    def gradient(self, coords: np.ndarray):
        # g = g(0) + H . dx
        dx = coords - self.centre
        new_g = self.grad + np.matmul(self.hess, dx)
        return new_g

    def hessian(self, coords: np.ndarray):
        # hessian is constant in second order expansion
        return self.hess


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
        init_trust: float = 0.2,
        line_search: bool = True,
        angle_thresh: Angle = Angle(5, units="deg"),
        *args,
        **kwargs,
    ):
        """
        Initialise a distance constrained optimiser. The pivot point
        is the point against which the distance is constrained. Optionally
        a linear search can be used to attempt to speed up convergence, but
        it does not always work.

        Args:
            init_trust: Initial trust radius in Angstrom
            pivot_point: Coordinates of the pivot point
            line_search: Whether to use linear search
            angle_thresh: An angle threshold above which linear search
                          will be rejected (in Degrees)
        """
        # todo replace init_alpha with init_trust in RFO later
        super().__init__(*args, init_alpha=init_trust, **kwargs)

        if not isinstance(pivot_point, CartesianCoordinates):
            raise NotImplementedError(
                "Internal coordinates are not implemented in distance"
                "constrained optimiser right now, please use Cartesian"
            )
        self._pivot = pivot_point
        self._do_line_search = bool(line_search)
        self._angle_thresh = Angle(angle_thresh, units="deg").to("radian")
        self._hessian_update_types = [BFGSUpdate, BofillUpdate]
        # todo replace later with bfgssr1update?

    def _initialise_run(self) -> None:
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._coords.make_hessian_positive_definite()
        self._update_gradient_and_energy()

    @property
    def converged(self) -> bool:
        if (
            self.rms_tangent_grad < self.gtol
            and self.last_energy_change < self.etol
        ):
            return True
        else:
            return False

    @property
    def rms_tangent_grad(self) -> GradientRMS:
        """
        Obtain the RMS of the gradient tangent to the distance
        vector between current coords and pivot point
        """
        grad = self._coords.g
        # unit vector in the direction of distance vector
        d_hat = self.dist_vec / np.linalg.norm(self.dist_vec)
        tangent_grad = grad - (grad.dot(d_hat)) * d_hat
        rms_grad = np.sqrt(np.mean(np.square(tangent_grad)))
        return GradientRMS(rms_grad)

    @property
    def dist_vec(self) -> np.ndarray:
        """
        Get the distance vector (p) between the current coordinates
        and the pivot point

        Returns:
            (np.ndarray):
        """
        return np.array(self._coords - self._pivot)

    @property
    def _target_dist(self):
        return np.linalg.norm(self.dist_vec)

    def _step(self) -> None:

        print(f"Distance between coords = {np.linalg.norm(self.dist_vec)}")
        self._coords.h = self._updated_h()

        if self.iteration > 1 and self._do_line_search:
            coords, grad = self._line_search_on_sphere()
        else:
            coords, grad = self._coords, self._coords.g

        step = self._get_lagrangian_step(coords, grad)

        step_size = np.linalg.norm(step)
        print(f"Taking a quasi-Newton step: {step_size:.3f} Angstrom")

        # the step is on the interpolated coordinates (if done)
        actual_step = (coords + step) - self._coords
        self._coords = self._coords + actual_step

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
        """
        from scipy.optimize import minimize

        taylor_pes = _TruncatedTaylor(coords, grad, self._coords.h)

        def step_size_constr(x):
            """step size must be <= trust radius"""
            step_est = x - self._coords
            # inequality constraint, should be > 0
            return self.alpha - np.linalg.norm(step_est)

        def lagrangian_constr(x):
            p = x - self._pivot
            return np.linalg.norm(p) - self._target_dist

        constrs = (
            {"type": "ineq", "fun": step_size_constr},
            {"type": "eq", "fun": lagrangian_constr},
        )

        res = minimize(
            fun=taylor_pes.value,
            x0=np.array(self._coords),
            method="slsqp",
            jac=taylor_pes.gradient,
            hess=taylor_pes.hessian,
            options={"maxiter": 2000},
            constraints=constrs,
        )

        if not res.success:
            raise RuntimeError

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
        # Eq (12) to (15) in J. Chem. Phys., 90, 1989, 2154
        # Notation follows the publication
        last_coords = self._history[-2]

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
        angle_change = abs(theta_prime - theta)
        if (
            angle_change > self._angle_thresh
            and abs(theta) > self._angle_thresh
        ) or (
            theta < 0  # extrapolating instead of interpolating
            and theta_prime < self._angle_thresh
        ):
            print("Linear interpolation step is unstable, skipping")
            return self._coords, self._coords.g

        p_interp = p_prime_prime * (
            np.cos(theta)
            - np.sin(theta) * np.cos(theta_prime) / np.sin(theta_prime)
        )
        p_interp += p_prime * (np.sin(theta) / np.sin(theta_prime))

        g_interp = last_coords.g * (1 - theta / theta_prime)
        g_interp += self._coords.g * (theta / theta_prime)

        x_interp = self._pivot + p_interp

        step_size = np.linalg.norm(x_interp - self._coords)
        print(f"Linear interpolation - step size: {step_size:.3f} Angstrom")

        return x_interp, g_interp


class DHSImagePair(ImagePair):
    """
    Image-pair used for Dewar-Healy-Stewart (DHS) method to
    find transition states
    """

    @property
    def ts_guess(self) -> Optional["Species"]:
        """
        In DHS method, the images can only rise in energy; therefore,
        the highest energy image is the ts_guess. If CI-NEB is run,
        then we can return CI-NEB as the final result.
        """
        tmp_spc = self._left_image.new_species(name="peak")

        if self._cineb_coords is not None:
            assert self._cineb_coords.e is not None
            tmp_spc.coordinates = self._cineb_coords
            return tmp_spc

        # NOTE: Even though the final points are probably the highest
        # this is not guaranteed, due to the probability of one end
        # jumping over the barrier. So we iterate through all coords

        energies = []
        for coord in self._total_history:
            energies.append(coord.e.to("Ha"))
        if any(x is None for x in energies):
            logger.error(
                "Energy values are missing in the trajectory of this"
                " image-pair. Unable to obtain transition state guess"
            )
            return None
        peak_idx = np.argmax(energies)
        tmp_spc.coordinates = self._total_history[peak_idx]


class DHS(BaseBracketMethod):
    """
    Dewar-Healy-Stewart method for finding transition states,
    from the reactant and product structures
    """

    def __init__(
        self,
        initial_species: "Species",
        final_species: "Species",
        reduction_factor: float = 0.05,
        **kwargs,
    ):
        """
        Dewar-Healy-Stewart method to find transition states.
        1) The order of initial_species/final_species does not matter
        and can be interchanged; 2) The reduction_factor is 0.05 or 5%
        by default, which is quite conservative, so may want to increase
        that if convergence is slow; 3) The distance tolerance should
        not be lowered any more than 1.0 Angstrom as DHS is unstable when
        the distance is low, and has a tendency for one image to jump
        over the barrier

        Args:
            initial_species: The "reactant" species

            final_species: The "product" species

            reduction_factor: The factor by which the distance is
                              decreased in each DHS step

        Keyword Args:

            maxiter: Maximum number of en/grad evaluations

            dist_tol: The distance tolerance at which DHS will
                      stop, values less than 1.0 Angstrom are not
                      recommended.

            gtol: Gradient tolerance for the optimiser micro-iterations
                  in DHS (Hartree/angstrom)
        """
        super().__init__(initial_species, final_species, **kwargs)

        # imgpair is only used for storing the points here
        self.imgpair = DHSImagePair(initial_species, final_species)
        self._reduction_fac = abs(float(reduction_factor))
        assert self._reduction_fac < 1.0

        # NOTE: In DHS the micro-iterations are done separately, in
        # an optimiser, so to keep track of the actual number of
        # en/grad calls, this local variable is used

        self._current_microiters: int = 0

        # todo replace reduction factor with something else

    @property
    def _method_name(self):
        return "DHS"

    def _initialise_run(self) -> None:
        """
        Initialise everything needed for the first DHS macro-iteration
        (Only energies are needed)
        """
        self.imgpair.update_one_img_mol_energy("left")
        self.imgpair.update_one_img_mol_energy("right")
        return None

    def calculate(self, method: "Method", n_cores: int) -> None:
        """
        Run the DHS calculation. Should only be called once!

        Args:
            method : Method used for calculating energy/gradients
            n_cores: Number of cores to use for calculation
        """
        self.imgpair.set_method_and_n_cores(method, n_cores)
        self._initialise_run()

        logger.info("Starting DHS method to find transition state")

        while not self.converged:

            if self.imgpair.left_coord.e < self.imgpair.right_coord.e:
                side = "left"
                pivot = self.imgpair.right_coord
            else:
                side = "right"
                pivot = self.imgpair.left_coord

            # take a step on the side with lower energy
            new_coord = self._dhs_step(side)

            # todo have to fix the total_iters
            # calculate the number of remaining maxiter to feed into optimiser
            curr_maxiter = self._maxiter - self._current_microiters
            if curr_maxiter == 0:
                break

            opt = DistanceConstrainedOptimiser(
                maxiter=curr_maxiter,
                gtol=self._gtol,
                etol=1.0e-3,  # seems like a reasonable etol
                pivot_point=pivot,
            )
            tmp_spc = self._species.copy()
            tmp_spc.coordinates = new_coord
            opt.run(tmp_spc, method)
            self._current_microiters += opt.iteration

            if not opt.converged:
                logger.error(
                    "Micro-iterations (optimisation) after a"
                    " DHS step did not converge, exiting"
                )
                break

            logger.info(
                "Successful optimization after DHS step, final RMS of"
                f" gradient = {opt.rms_tangent_grad:.6f} Ha/angstrom"
            )

            # put results back into imagepair
            if side == "left":
                self.imgpair.left_coord = opt.final_coordinates
            elif side == "right":
                self.imgpair.right_coord = opt.final_coordinates
            else:
                raise ImgPairSideError()

            if self.imgpair.has_jumped_over_barrier(side):
                logger.warning(
                    "One image has jumped over the other image"
                    " while running DHS optimisation. This"
                    " indicates that the distance between images"
                    " is quite close, so DHS cannot proceed even"
                    " though the distance criteria is not met"
                )
                break

            self._log_convergence()

        # exited loop, print final message
        logger.info(
            f"Finished DHS procedure in {self._macro_iter} macro-"
            f"iterations consisting of {self._current_microiters}"
            f" micro-iterations (optimiser steps). DHS is "
            f"{'converged' if self.converged else 'not converged'}"
        )

        return None

    def _log_convergence(self) -> None:
        logger.info(
            f"Macro-iteration #{self._macro_iter}: Distance = "
            f"{self.imgpair.dist:.4f}; Energy (initial species) = "
            f"{self.imgpair.left_coord.e:.6f}; Energy (final species) = "
            f"{self.imgpair.right_coord.e:.6f}"
        )
        return None

    @property
    def _macro_iter(self):
        """Total number of DHS steps taken so far"""
        # ImagePair only stores the converged coordinates, which
        # is equal to the number of macro-iterations (DHS steps)
        return self.imgpair.total_iters

    def _dhs_step(self, side: str) -> CartesianCoordinates:
        """
        Take a DHS step, on the side requested, along the distance
        vector between the two images

        Args:
            side (str):

        Returns:
            (CartesianCoordinates): New predicted coordinates for that side
        """
        # take a DHS step by minimizing the distance by factor
        new_dist = (1 - self._reduction_fac) * self.imgpair.dist
        dist_vec = self.imgpair.dist_vec
        step = dist_vec * self._reduction_fac  # ??

        if side == "left":
            new_coord = self.imgpair.left_coord - step
        elif side == "right":
            new_coord = self.imgpair.right_coord + step
        else:
            raise ImgPairSideError()

        logger.info(
            f"DHS step on {side} image: setting distance to {new_dist:.4f}"
        )
        return new_coord
