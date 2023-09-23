"""
Improved Elastic Image Pair method for finding transition states.

References:

[1] Y. Liu, H. Qi, M. Lei, J. Chem. Theory Comput., 2023, 19, 2410-2417
"""
from typing import Union, Optional, Tuple, List, TYPE_CHECKING
import numpy as np
from autode.methods import get_lmethod
from autode.bracket.base import BaseBracketMethod
from autode.bracket.dhs import TruncatedTaylor
from autode.neb import NEB
from autode.path.interpolation import CubicPathSpline
from autode.bracket.imagepair import EuclideanImagePair
from autode.opt.coordinates import CartesianCoordinates
from autode.values import Distance, GradientRMS, PotentialEnergy
from autode.utils import ProcessPool
from autode.log import logger

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method

_interp_image_density = 1.0  # per Angstrom for initial interpolation


def _calculate_low_sp_energy_for_species(
    species: "Species", method: "Method", n_cores: int
) -> "PotentialEnergy":
    """
    Convenience function to calculate the energy for a given species

    Args:
        species (Species): The species object
        method (Method): The method (low_sp keywords will be used)
        n_cores (int): The number of cores

    Returns:
        (PotentialEnergy): The single point energy of the species
    """
    from autode import Calculation

    sp_calc = Calculation(
        name=f"{species.name}_sp",
        molecule=species,
        method=method,
        keywords=method.keywords.low_sp,  # NOTE: We use low_sp
        n_cores=n_cores,
    )

    sp_calc.run()
    sp_calc.clean_up(force=True, everything=True)

    assert species.energy is not None
    return species.energy


def _parallel_calc_energies(
    points: List["Species"], method: "Method", n_cores: int
) -> List["PotentialEnergy"]:
    """
    Calculate the single point energies on a list of species with
    parallel runs

    Args:
        points (list[Species]): A list of the species
        method (Method): The method (low_sp keywords will be used)
        n_cores (int): Total number of cores for all calculations

    Returns:
        (list[PotentialEnergy]): List of energies in order
    """
    n_cores_per_pp = max(n_cores // len(points), 1)
    n_procs = min(n_cores, len(points))

    with ProcessPool(max_workers=n_procs) as pool:
        jobs = [
            pool.submit(
                _calculate_low_sp_energy_for_species,
                species=point,
                method=method,
                n_cores=n_cores_per_pp,
            )
            for point in points
        ]

        energies = [job.result() for job in jobs]

    return energies


class ElasticImagePair(EuclideanImagePair):
    """
    This image-pair used for the Elastic Image Pair calculation. The
    geometries after every macro-iteration are stored
    """

    @property
    def last_left_step_size(self) -> float:
        """The last step size on the left image"""
        return np.linalg.norm(self.left_coords - self._left_history[-2])

    @property
    def last_right_step_size(self):
        """The last step size on the right image"""
        return np.linalg.norm(self.right_coords - self._right_history[-2])

    @property
    def ts_guess(self) -> Optional["Species"]:
        """
        Obtain the TS guess from the i-EIP image pair. The midpoint
        between the two converged images is considered the TS guess

        Returns:
            (Species|None): The ts guess species, if images are available
        """
        # account for initial redistribution
        if self.total_iters <= 2:
            return None

        tmp_spc = self._left_image.new_species(name="peak")
        midpt_coords = np.array(self.left_coords + self.right_coords) / 2
        tmp_spc.coordinates = midpt_coords
        return tmp_spc

    @property
    def perp_rms_gs(self) -> Tuple[GradientRMS, GradientRMS]:
        """
        The RMS norms of perpendicular gradient component for left
        and right image, in order. The parallel component against
        the distance vector is projected out.

        Returns:
            (tuple[GradientRMS, GradientRMS]):
        """
        perp_gradients = []
        d_hat = self.dist_vec / np.linalg.norm(self.dist_vec)
        for coord in [self.left_coords, self.right_coords]:
            parall_g = d_hat * np.dot(d_hat, coord.g)
            perp_g = coord.g - parall_g
            rms_perp_g = np.sqrt(np.mean(np.square(perp_g)))
            perp_gradients.append(GradientRMS(rms_perp_g))

        return perp_gradients[0], perp_gradients[1]

    def redistribute_imagepair(
        self,
        ll_neb_interp: bool = True,
        interp_fraction: float = 1 / 4,
    ):
        """
        Redistribute the image pair by running a NEB calculation at lmethod or
        use only IDPP interpolation, and then fitting a cubic spline on energy
        calculated by the method (low_sp keywords). It generates the image pair
        on both sides of the peak on the fitted spline, with a distance of
        interp_fraction * total path distance on either side.

        Args:
            ll_neb_interp (bool): Whether to optimise the interpolated path with
                                  NEB at lmethod for the interpolation.

            interp_fraction (float): Fraction of total interpolated path distance
                                     that will be used to generate the image pair
                                     on either side of the interpolated TS
        """
        # Use at least 5 images for interpolation
        n_images = int(_interp_image_density * self.dist - 1)
        n_images = max(n_images, 5 + 2)

        interp = NEB.from_end_points(
            self._left_image.copy(), self._right_image.copy(), n_images
        )
        if ll_neb_interp:
            interp.calculate(method=get_lmethod(), n_cores=self._n_cores)

        # Only calc intermediate images, initial and final already have energies
        path_points = interp.images[1:-1]
        # Get energies
        assert self.left_coords.e and self.right_coords.e
        assert self._method is not None and self._n_cores is not None
        path_energies = _parallel_calc_energies(
            path_points, method=self._method, n_cores=self._n_cores
        )
        energies = [self.left_coords.e] + path_energies + [self.right_coords.e]
        logger.info(
            f"Fitting parametric spline on {len(interp.images)} points"
        )

        # NOTE: Here we are fitting a parametric spline, with the parameter
        # being the path length along approx. rxn coordinate and target being all
        # coordinates *and* energy at the points
        path_spline = CubicPathSpline.from_species_list(interp.images)
        path_spline.fit_energies(energies)
        peak_x = path_spline.energy_peak()
        if peak_x is None:
            raise RuntimeError(
                "The fitted spline does not have a peak! Unable to proceed"
            )
        # Check the peak is not at the beginning or end
        assert 0.01 < peak_x < 0.99
        # convert to integrated arc lengths
        peak_pos = path_spline.path_integral(0, peak_x)
        path_length = path_spline.path_integral(0, 1)

        # Generate new coordinates a fraction (default 1/4) of total distance
        # on each side of the peak (interpolated TS)
        left_span = peak_pos - interp_fraction * path_length
        if left_span <= 0.01:
            l_point = 0.0
        else:
            l_point = path_spline.integrate_upto_length(
                span=left_span,
            )
        r_point = path_spline.integrate_upto_length(
            span=peak_pos + interp_fraction * path_length,
        )
        if r_point > 1:
            r_point = 1
        self.left_coords = CartesianCoordinates(path_spline.coords_at(l_point))
        self.right_coords = CartesianCoordinates(
            path_spline.coords_at(r_point)
        )
        return None


class IEIPMicroImagePair(EuclideanImagePair):
    """
    Class to carry out the micro-iterations for the i-EIP
    method
    """

    def __init__(
        self,
        species: "Species",
        left_coords: CartesianCoordinates,
        right_coords: CartesianCoordinates,
        micro_step_size: Union[Distance, float],
        target_dist: Union[Distance, float],
    ):
        """
        Initialise an image-pair for i-EIP micro-iterations

        Args:
            species (Species): The species object
            left_coords (CartesianCoordinates): Coordinates of left image, must
                                        have defined energy, gradient and hessian
            right_coords (CartesianCoordinates): Coordinates of right image, must
                                        have defined energy, gradient and hessian
            micro_step_size (Distance|float): Step size for each micro-iteration
            target_dist (Distance|float): Target distance for image-pair
        """
        # dummy species are required for superclass init
        left_image, right_image = species.copy(), species.copy()
        left_image.coordinates = left_coords
        right_image.coordinates = right_coords
        assert isinstance(left_coords, CartesianCoordinates)
        assert isinstance(right_coords, CartesianCoordinates)
        super().__init__(left_image=left_image, right_image=right_image)

        # generate the Taylor expansion surface from gradient and hessian
        assert left_coords.g is not None and left_coords.h is not None
        assert right_coords.g is not None and right_coords.h is not None
        self._left_taylor_pes = TruncatedTaylor(
            left_coords, left_coords.g, left_coords.h
        )
        self._right_taylor_pes = TruncatedTaylor(
            right_coords, right_coords.g, right_coords.h
        )
        self._micro_step = float(Distance(micro_step_size, "ang"))
        self._target_dist = float(Distance(target_dist, "ang"))

    @property
    def ts_guess(self) -> Optional["Species"]:
        """This class does not implement TS guess"""
        raise NotImplementedError

    def update_both_img_engrad(self) -> None:
        """
        Update the energy and gradient from the Taylor surface
        """
        self.left_coords.e = PotentialEnergy(
            self._left_taylor_pes.value(self.left_coords)
        )
        self.left_coords.g = self._left_taylor_pes.gradient(self.left_coords)
        self.right_coords.e = PotentialEnergy(
            self._right_taylor_pes.value(self.right_coords)
        )
        self.right_coords.g = self._right_taylor_pes.gradient(
            self.right_coords
        )
        return None

    @property
    def n_hat(self):
        """
        Unit vector pointing from right to left image, the vector is
        reverse (negative) of how it is defined in publication [1]
        """
        return self.dist_vec / np.linalg.norm(self.dist_vec)

    def _get_perpendicular_micro_steps(self) -> List[np.ndarray]:
        """
        Obtain the perpendicular displacement for one i-EIP micro-iteration,
        minimises the energy in the direction perpendicular to the distance
        vector connecting the image pair

        Returns:
            (list[np.ndarray]): A list of steps for left and right
                                image, in order
        """
        assert self.left_coords.g is not None
        assert self.right_coords.g is not None
        steps = []
        for coord in [self.left_coords, self.right_coords]:
            force = -coord.g  # type: ignore
            force_parall = self.n_hat * np.dot(force, self.n_hat)
            force_perp = force - force_parall
            delta_x_perp = force_perp / np.linalg.norm(force_perp)
            delta_x_perp *= min(np.linalg.norm(force_perp), self._micro_step)
            steps.append(delta_x_perp)

        return steps

    def _get_energy_micro_steps(self) -> List[np.ndarray]:
        """
        Obtain the energy based displacement term for one i-EIP
        micro-iteration. This term minimises the energy difference
        between the two images

        Returns:
            (list[np.ndarray]): A list of steps for the left and right
                                image, in order
        """
        # NOTE: The sign is flipped here, because distance vector is
        # defined in the opposite direction i.e. left - right
        assert self.left_coords.e and self.right_coords.e
        f_de = (self.left_coords.e - self.right_coords.e) / float(self.dist)
        f_de = self.n_hat * float(f_de)
        delta_x_e = f_de / np.linalg.norm(f_de)
        delta_x_e *= min(np.linalg.norm(f_de), self._micro_step)
        return [delta_x_e, delta_x_e]

    def _get_distance_micro_steps(self):
        """
        Obtain the displacement term that controls the distance between
        the two images. This term moves the images so that their distance
        can be closer to the target distance in the current macro-iteration

        Returns:
            (list[np.ndarray]): A list of steps for the left and right
                                images, in order
        """
        # NOTE: The factor k that appears in eqn.(1) of the i-EIP paper
        # has been absorbed into the term in this function (i.e. the function
        # returns the displacements with the proper sign)
        f_l = 2 * (self.dist - self._target_dist) / self.dist
        f_l = -self.dist_vec * f_l
        delta_x_l = f_l / np.linalg.norm(f_l)
        delta_x_l *= min(np.linalg.norm(f_l), self._micro_step)
        return [delta_x_l, -delta_x_l]

    def take_micro_step(self) -> None:
        """
        Take a single i-EIP micro-iteration step (which is a sum of the
        perpendicular, energy and distance terms)
        """
        perp_steps = self._get_perpendicular_micro_steps()
        energy_steps = self._get_energy_micro_steps()
        dist_steps = self._get_distance_micro_steps()

        # sum the micro-iteration step components
        left_step = perp_steps[0] + energy_steps[0] + dist_steps[0]
        right_step = perp_steps[1] + energy_steps[1] + dist_steps[1]

        # scale the steps to have the selected micro step size
        left_step /= np.linalg.norm(left_step)
        left_step *= min(np.linalg.norm(left_step), self._micro_step)
        right_step /= np.linalg.norm(right_step)
        right_step *= min(np.linalg.norm(right_step), self._micro_step)

        self.left_coords = self.left_coords + left_step
        self.right_coords = self.right_coords + right_step
        self._flush_old_matrices()
        return None

    def _flush_old_matrices(self):
        """The old gradient matrices should be removed to save memory"""
        if self.total_iters / 2 > 2:
            self._left_history[-3].g = None
            self._right_history[-3].g = None

    @property
    def max_displacement(self) -> float:
        """
        The maximum displacement on either of the images compared
        to the starting point (for the current macro-iteration)

        Returns:
            (float): The max displacement
        """
        left_displ = np.linalg.norm(self.left_coords - self._left_history[0])
        right_displ = np.linalg.norm(
            self.right_coords - self._right_history[0]
        )
        return max(left_displ, right_displ)

    @property
    def n_micro_steps(self):
        """
        Total number of micro-iterations performed on this image pair
        """
        return int(self.total_iters / 2)


class IEIP(BaseBracketMethod):
    """
    Improved Elastic Image Pair Method (i-EIP). It performs an initial
    interpolation followed by spline fitting to redistribute the image
    pair close to the interpolated TS. Then, micro-iterations are performed
    to move the images closer while maintaining the distance
    """

    def __init__(
        self,
        initial_species: "Species",
        final_species: "Species",
        micro_step_size: Union[Distance, float] = Distance(1.5e-5, "ang"),
        max_micro_per_macro: int = 2000,
        max_macro_step: Union[Distance, float] = Distance(0.15, "ang"),
        use_ll_neb_interp: bool = True,
        interp_fraction: float = 1 / 4,
        dist_tol: Union[Distance, float] = Distance(0.3, "ang"),
        gtol: Union[GradientRMS, float] = GradientRMS(0.02, "Ha/ang"),
        maxiter: int = 200,
        **kwargs,
    ):
        """
        Initialise an i-EIP calculation from the initial (reactant) and
        final (product) species. Every macro-iteration consists of two
        gradient evaluations on both images, therefore, the total number
        of gradient evaluations would be 2 * maxiter. Increase the
        interp_fraction argument to start the calculation closer to the
        reactant and product, and move ahead less with the initial
        interpolation.

        Args:
            initial_species: The "reactant" species

            final_species: The "product" species

            micro_step_size: The step size for every micro-iteration

            max_micro_per_macro: The maximum number of micro-iterations
                                per macro-iteration

            max_macro_step: The maximum step size for one macro-iteration

            use_ll_neb_interp: Whether to use lmethod NEB for the
                            initial interpolation instead of only IDPP
                            interpolation

            interp_fraction: Generate image pair on both sides of the
                            interpolated peak with this fraction of the
                            total interpolated path length (default 1/4)

            dist_tol: The Euclidean distance tolerance (between images)
                      for convergence

            gtol: The tolerance for perpendicular gradient RMS norm

            maxiter: For i-EIP maxiter is the maximum number of macro-iterations

        """
        assert (
            "cineb_at_conv" not in kwargs.keys()
        ), "CI-NEB refinement is not available for i-EIP method!"

        super().__init__(
            initial_species,
            final_species,
            gtol=gtol,
            dist_tol=dist_tol,
            maxiter=maxiter,
            **kwargs,
        )

        self.imgpair: ElasticImagePair = ElasticImagePair(
            initial_species, final_species
        )
        self._micro_step_size = Distance(micro_step_size, "ang")
        assert self._micro_step_size > 0
        self._max_micro_per = abs(int(max_micro_per_macro))
        self._max_macro_step = Distance(max_macro_step, "ang")
        assert self._max_macro_step > 0
        self._ll_neb_interp = bool(use_ll_neb_interp)
        self._interp_frac = float(interp_fraction)
        assert 0 < interp_fraction < 1

        # NOTE: In EIP the microiters are done separately in a throwaway
        # imagepair object, so a variable is required to keep track
        self._current_microiters: int = 0

        self._target_dist: Optional[float] = None
        self._target_rms_g: Optional[float] = None

    @property
    def _micro_iter(self) -> int:
        """
        Total number of micro-iterations. For i-EIP each micro-iteration
        is on both of the images simultaneously
        """
        return self._current_microiters

    @_micro_iter.setter
    def _micro_iter(self, value):
        """Set the total number of micro-iterations"""
        self._current_microiters = int(value)

    @property
    def _macro_iter(self) -> int:
        """Total number of macro-iterations taken"""
        # minus 1 due to first redistribution
        return int(self.imgpair.total_iters / 2) - 1

    @property
    def converged(self) -> bool:
        """Is the i-EIP method converged"""
        # NOTE: Original publication recommends also checking overlap
        # of image-pair mode with Hessian eigenvalue for convergence, but
        # Hessian is expensive, so we use simpler check
        return self.imgpair.dist < self._dist_tol and all(
            rms_grad <= self._gtol for rms_grad in self.imgpair.perp_rms_gs
        )

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        """Whether it has exceeded the number of maximum iterations"""
        if self._macro_iter >= self._maxiter:
            logger.error(
                f"Reached the maximum number of micro-iterations "
                f"*{self._maxiter}"
            )
            return True
        else:
            return False

    def _initialise_run(self) -> None:
        """
        Initialise the i-EIP calculation by redistributing the
        image pair and then estimating a low level hessian
        """
        self.imgpair.update_both_img_engrad()
        self.imgpair.redistribute_imagepair(
            self._ll_neb_interp, self._interp_frac
        )
        self.imgpair.update_both_img_engrad()
        self.imgpair.update_both_img_hessian_by_calc()
        self._target_dist = self.imgpair.dist
        self._target_rms_g = (
            min(max(self.imgpair.dist / self._dist_tol, 1), 2) * self._gtol
        )
        return None

    def _step(self) -> None:
        """
        Take one EIP macro-iteration step and store the new coordinates
        in history and update the energies and gradients
        """
        self._update_target_distance_and_force()

        # Turn off logging for micro-iterations
        logger.disabled = True
        assert self._target_dist is not None
        micro_imgpair = IEIPMicroImagePair(
            species=self._species,
            left_coords=self.imgpair.left_coords,
            right_coords=self.imgpair.right_coords,
            micro_step_size=self._micro_step_size,
            target_dist=self._target_dist,
        )

        while not (
            micro_imgpair.n_micro_steps >= self._max_micro_per
            or micro_imgpair.max_displacement > self._max_macro_step
        ):
            micro_imgpair.update_both_img_engrad()
            micro_imgpair.take_micro_step()
            self._micro_iter += 1
        logger.disabled = False

        self.imgpair.left_coords = micro_imgpair.left_coords
        self.imgpair.right_coords = micro_imgpair.right_coords
        self.imgpair.update_both_img_engrad()
        self.imgpair.update_both_img_hessian_by_formula()

        logger.info(
            f"Completed one i-EIP macro-iteration with "
            f"{micro_imgpair.n_micro_steps} micro-iterations; maximum "
            f"image displacement = {micro_imgpair.max_displacement:.3f}.\n"
            f"Left image step: {self.imgpair.last_left_step_size:.3f}, "
            f"Right image step: {self.imgpair.last_right_step_size:.3f}"
        )
        return None

    def _update_target_distance_and_force(self):
        """
        Update the target distance tolerance and the RMS gradients
        for the current macro-iteration
        """
        # only update if target RMS force and distance has been reached
        if not all(
            rms_grad <= self._target_rms_g
            for rms_grad in self.imgpair.perp_rms_gs
        ):
            return None

        # NOTE: target distance near the end of optimisation
        # must be slighly lower than the set dist_tol, otherwise
        # it will never converge (as it won't go below dist_tol)
        self._target_dist = max(
            0.9 * self.imgpair.dist,
            self._dist_tol - 0.015,
        )

        self._target_rms_g = (
            min(max(self.imgpair.dist / self._dist_tol, 1), 2) * self._gtol
        )

        logger.info(
            f"Updating target distance to {self._target_dist:.3f} Å"
            f" and updating target RMS gradient to "
            f"{self._target_rms_g:.3f} Ha/Å"
        )
        return None
