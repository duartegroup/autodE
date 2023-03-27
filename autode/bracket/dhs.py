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
        """
        Second-order Taylor expansion around a point

        Args:
            centre (OptCoordinates|np.ndarray): The coordinate point
            grad (np.ndarray): Gradient at that point
            hess (np.ndarray): Hessian at that point
        """
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
        """Energy (or relative energy if point did not have energy)"""
        # E = E(0) + g^T . dx + 0.5 * dx^T. H. dx
        dx = (coords - self.centre).flatten()
        new_e = self.en + np.dot(self.grad, dx)
        new_e += 0.5 * np.linalg.multi_dot((dx, self.hess, dx))
        return new_e

    def gradient(self, coords: np.ndarray):
        """Gradient at supplied coordinate"""
        # g = g(0) + H . dx
        dx = (coords - self.centre).flatten()
        new_g = self.grad + np.matmul(self.hess, dx)
        return new_g

    def hessian(self, coords: np.ndarray):
        """Hessian at supplied coordinates"""
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
        self._target_dist = None

        self._hessian_update_types = [BFGSUpdate, BofillUpdate]
        # todo replace later with bfgssr1update?

    def _initialise_run(self) -> None:
        """Initialise self._coords, gradient and hessian"""
        self._coords = CartesianCoordinates(self._species.coordinates)
        self._target_dist = np.linalg.norm(self.dist_vec)
        self._coords.update_h_from_cart_h(self._low_level_cart_hessian)
        self._coords.make_hessian_positive_definite()
        self._update_gradient_and_energy()

    @property
    def converged(self) -> bool:
        """Has the optimisation converged"""
        # The tangential gradient should be close to zero
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

        # NOTE: Since the linear interpolation should produce a point
        # in the vicinity of the last two points, it seems reasonable to
        # also use the hessian from the last point in the case of linear
        # interpolation being done
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
            # hess=taylor_pes.hessian,
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
        then that result is returned instead
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
        peak_coords = self._total_history[np.argmax(energies)]
        tmp_spc.coordinates = peak_coords
        tmp_spc.energy = peak_coords.e
        tmp_spc.gradient = peak_coords.g
        return tmp_spc

    def has_jumped_over_barrier(self, side: str) -> bool:
        """
        The simplest test would be to check the distances between
        the last two points, and the newly added points. If the newly
        added point does not lie between the last two points (in
        Euclidean distance) then the point has probably jumped over
        the barrier (This is not a strict/rigorous test, but very cheap)

        Args:
            side: side of the newly added point

        Returns:
            (bool): whether the point has probably jumped over
        """
        if side == "left":
            last_coord = self._left_history[-1]
            other_coord = self.right_coord
        elif side == "right":
            last_coord = self._right_history[-1]
            other_coord = self.left_coord
        else:
            raise ImgPairSideError()

        new_coord = self.get_coord_by_side(side)

        # We assume the next point will lie between the
        # last two images on both side. If the current coord
        # is further from last coord on the same side than the
        # opposite image, then it has jumped over
        dist_to_last = np.linalg.norm(new_coord - last_coord)
        dist_before_step = np.linalg.norm(other_coord - last_coord)

        if dist_to_last >= dist_before_step:
            return True
        else:
            return False

    def get_last_step_by_side(self, side: str) -> CartesianCoordinates:
        _, _, hist, _ = self._get_img_by_side(side)
        return hist[-1] - hist[-2]


class DHS(BaseBracketMethod):
    """
    Dewar-Healy-Stewart method for finding transition states,
    from the reactant and product structures
    """

    def __init__(
        self,
        initial_species: "Species",
        final_species: "Species",
        step_size: Distance = Distance(0.1, "ang"),
        **kwargs,
    ):
        """
        Dewar-Healy-Stewart method to find transition states.

        1) The step size is 0.1 which is quite conservative, so may want
        to increase that if convergence is slow; 2) The distance tolerance
        should not be lowered any more than 1.0 Angstrom as DHS is unstable
        when the distance is low, and there is a tendency for one image to
        jump over the barrier

        Args:
            initial_species: The "reactant" species

            final_species: The "product" species

            step_size: The size of the DHS step taken along
                        the linear path between reactant and
                        product

        Keyword Args:

            maxiter: Maximum number of en/grad evaluations

            dist_tol: The distance tolerance at which DHS will
                      stop, values less than 1.0 Angstrom are not
                      recommended.

            gtol: Gradient tolerance for the optimiser micro-iterations
                  in DHS (Hartree/angstrom)

            cineb_at_conv: Whether to run CI-NEB calculation from the end
                           points after the DHS is converged
        """
        super().__init__(initial_species, final_species, **kwargs)

        # imgpair is only used for storing the points here
        self.imgpair = DHSImagePair(initial_species, final_species)

        self._step_size = Distance(abs(step_size), "ang")
        assert self._step_size < self.imgpair.dist

        # NOTE: In DHS the micro-iterations are done separately, in
        # an optimiser, so to keep track of the actual number of
        # en/grad calls, this local variable is used

        self._current_microiters: int = 0

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
            new_coord = self._get_dhs_step(side)

            # calculate the number of remaining maxiter to feed into optimiser
            curr_maxiter = self._maxiter - self._current_microiters
            if curr_maxiter <= 0:
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

    def _get_dhs_step(self, side: str) -> CartesianCoordinates:
        """
        Take a DHS step, on the side requested, along the distance
        vector between the two images, and return the new coordinates
        after taking the step

        Args:
            side (str):

        Returns:
            (CartesianCoordinates): New predicted coordinates for that side
        """
        # take a DHS step of the size given
        dist_vec = self.imgpair.dist_vec
        dhs_step = dist_vec * (self._step_size / self.imgpair.dist)

        if side == "left":
            new_coord = self.imgpair.left_coord - dhs_step
        elif side == "right":
            new_coord = self.imgpair.right_coord + dhs_step
            # todo make a function that will put into side
        else:
            raise ImgPairSideError()

        logger.info(
            f"DHS step on {side} image: taking a step of"
            f" size {self._step_size:.4f}"
        )
        return new_coord


class HybridGSDHS(DHS):
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
        Keyword Args:
            gs_mix (float): Represents the percentage of mixing of the
                            Growing String step with the DHS step. 0.3
                            means 0.3 * GS_step + (1-0.3) * DHS_step
                            It is recommended to set this to any fraction
                            lower than 0.5
        """
        super().__init__(*args, **kwargs)

        self._gs_mix = float(gs_mix)
        assert 0.0 < self._gs_mix < 1.0, "Mixing factor must be 0 < fac < 1"

    def _get_dhs_step(self, side: str) -> CartesianCoordinates:
        """
        Take a mixed DHS and GS step (interpolates between the two
        vectors) in the given ratio, and then return the new
        coordinates after taking the step

        Args:
            side (str):

        Returns:
            (CartesianCoordinates): New predicted coordinates for that side
        """
        # obtain the DHS step
        dist_vec = self.imgpair.dist_vec
        dhs_step = dist_vec * (self._step_size / self.imgpair.dist)

        gs_step = self.imgpair.get_last_step_by_side(side)
        old_coord = self.imgpair.get_coord_by_side(side)

        # todo should I rescale the growing string step?
        if side == "left":
            new_coord = (
                self.imgpair.left_coord
                - (1 - self._gs_mix) * dhs_step + self._gs_mix * gs_step
            )
        elif side == "right":
            new_coord = (
                self.imgpair.right_coord
                + (1 - self._gs_mix) * dhs_step + self._gs_mix * gs_step
            )
        else:
            raise ImgPairSideError()

        step_size = np.linalg.norm(new_coord - old_coord)
        logger.info(
            f"DHS-GS step on {side} image: taking a step "
            f"of size {step_size:.4f}"
        )

        return new_coord
