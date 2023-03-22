"""
Dewar-Healy-Stewart Method for finding transition states

As described in J. Chem. Soc. Farady Trans. 2, 1984, 80, 227-233
"""

import os
from typing import Tuple, Union, Optional
import numpy as np
from matplotlib import pyplot as plt

from autode.values import Distance, Angle, GradientRMS
from autode.units import ang as angstrom
from autode.bracket.imagepair import ImagePair, ImgPairSideError
from autode.methods import get_lmethod
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate, BFGSUpdate
from autode.opt.optimisers.base import _OptimiserHistory
from autode.input_output import atoms_to_xyz_file
from autode.opt.optimisers import RFOptimiser

from autode.log import logger
from autode.config import Config

import autode.species.species
import autode.wrappers.methods


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
        target_dist: Distance,
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
        self._target_dist = Distance(target_dist, units="ang")
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
            (np.ndarray): Step in cartesian (or mw-cartesian)
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
            method="SLSQP",
            jac=taylor_pes.gradient,
            hess=taylor_pes.hessian,
            options={"disp": True},
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
    An image-pair that defines the distance between two images
    as the Euclidean distance (square of root of sum of the
    squares of deviations in Cartesian coordinates)
    """
    # todo remove this once done

    @property
    def dist_vec(self) -> np.ndarray:
        """
        The distance vector in Cartesian coordinates, pointing
        to right_image from left_image
        """
        return np.array(
            self.left_coord.to("cart") - self.right_coord.to("cart")
        )

    @property
    def euclid_dist(self) -> Distance:
        """
        The Euclidean distance in Cartesian coordinates between the images

        Returns:
            (Distance): Distance in angstrom
        """
        return Distance(np.linalg.norm(self.dist_vec), "ang")

    def get_one_img_perp_grad(self, side: str) -> np.ndarray:
        """
        Get the gradient perpendicular to the distance vector
        between the two images of the image-pair, for one
        image

        Returns:
            (np.ndarray): The perpendicular component
        """
        dist_vec = self.dist_vec
        # get unit vector in that direction
        unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
        _, coord, _, _ = self._get_img_by_side(side)
        # project the cartesian gradient towards the distance vector
        proj_cart_grad = unit_dist_vec * coord.to("cart").g.dot(unit_dist_vec)
        # gradient component perpendicular to distance vector
        perp_cart_grad = np.array(coord.to("cart").g - proj_cart_grad)

        if isinstance(coord, CartesianCoordinates):
            # no need to convert for cartesian coords
            return perp_cart_grad

        else:
            raise NotImplementedError("Please use Cartesian coordinates!")
        # todo remove once coordinates are sorted


class DHS:
    """
    Dewar-Healy-Stewart method for finding transition states,
    from the reactant and product structures
    """

    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        maxiter: int = 300,
        reduction_factor: float = 0.05,
        dist_tol: Union[Distance, float] = Distance(1.0, "ang"),
        gtol: Optional[GradientRMS] = GradientRMS(1.e-3, "ha/ang"),
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

            maxiter: Maximum number of en/grad evaluations

            reduction_factor: The factor by which the distance is
                              decreased in each DHS step

            dist_tol: The distance tolerance at which DHS will
                      stop, values less than 1.0 Angstrom are not
                      recommended.

            gtol: Gradient tolerance for the optimiser micro-iterations
                  in DHS
        """
        # imgpair is only used for storing the points here
        self.imgpair = ImagePair(initial_species, final_species)
        self._species = initial_species.copy()  # just hold the species
        self._reduction_fac = abs(float(reduction_factor))
        assert self._reduction_fac < 1.0

        self._maxiter = abs(int(maxiter))
        self._dist_tol = Distance(dist_tol, "ang")
        self._gtol = GradientRMS(gtol, "ha/ang")

        # these only hold the coords after finishing optimisation
        # for each DHS step. Put the initial coordinates here
        self._initial_species_hist = _OptimiserHistory()
        self._initial_species_hist.append(self.imgpair.left_coord)
        self._final_species_hist = _OptimiserHistory()
        self._final_species_hist.append(self.imgpair.right_coord)
        # todo clean up imgpair and remove coord stuff
        # todo remove this when GS2 corrector is done

    @property
    def converged(self) -> bool:
        """Is DHS converged to the desired distance tolerance?"""
        if self.imgpair.dist < self._dist_tol:
            return True
        else:
            return False

    def _initialise_run(self) -> None:
        """
        Initialise everything needed for the first DHS macro-iteration
        (Only energies are needed)
        """
        self.imgpair.update_one_img_mol_energy("left")
        self.imgpair.update_one_img_mol_energy("right")
        return None

    def calculate(
        self, method: autode.wrappers.methods.Method, n_cores: int
    ) -> None:
        """
        Run the DHS calculation. Should only be called once!

        Args:
            method : Method used for calculating energy/gradients
            n_cores: Number of cores to use for calculation
        """
        self.imgpair.set_method_and_n_cores(method, n_cores)
        self._initialise_run()

        logger.info("Starting DHS optimisation")

        while not self.converged:

            if self.imgpair.left_coord.e < self.imgpair.right_coord.e:
                side = "left"
                pivot = self.imgpair.right_coord
            else:
                side = "right"
                pivot = self.imgpair.left_coord

            # take a step on the side with lower energy
            self._step(side)
            dist = self.imgpair.dist
            coord0 = np.array(self.imgpair.get_coord_by_side(side))

            # calculate the number of remaining maxiter to feed into optimiser
            curr_maxiter = self._maxiter - self.imgpair.total_iters
            if curr_maxiter == 0:
                break

            opt = DistanceConstrainedOptimiser(
                maxiter=curr_maxiter,
                gtol=1e-3,
                etol=1e-3,
                pivot_point=pivot,
                target_dist=dist,
            )
            self._species.coordinates = coord0
            opt.run(self._species, method)

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
            # todo has jumped over the barrier should be on other side

            # if optimisation succeeded
            """new_coord = self.imgpair.get_coord_by_side(side)
            # check if it has jumped over the barrier
            # todo fix this function
            if self._has_jumped_over_barrier(new_coord, side):
                logger.warning(
                    "One image has jumped over the other image"
                    " while running DHS optimisation. This"
                    " indicates that the distance between images"
                    " is quite close, so DHS cannot proceed even"
                    " though the distance criteria is not met"
                )
                break"""

            self._log_convergence()

        # exited loop, print final message
        logger.info(
            f"Finished DHS procedure in {self.macro_iter} macro-"
            f"iterations consisting of {self.imgpair.total_iters}"
            f" micro-iterations (optimiser steps). DHS is "
            f"{'converged' if self.converged else 'not converged'}"
        )

        return None

    def _log_convergence(self) -> None:
        logger.info(
            f"Macro-iteration #{self.macro_iter}: Distance = "
            f"{self.imgpair.dist:.4f}; Energy (initial species) = "
            f"{self.imgpair.left_coord.e:.6f}; Energy (final species) = "
            f"{self.imgpair.right_coord.e:.6f}"
        )
        return None

    def run(self) -> None:
        """
        Runs the DHS calculation with the default low-level
        method, and the number of cores from currently set Config,
        then writes the trajectories and energy plot, then prints
        the highest energy point as xyz file
        """
        lmethod = get_lmethod()
        self.calculate(method=lmethod, n_cores=Config.n_cores)
        self.write_trajectories()
        self.plot_energies()
        self.get_peak_species().print_xyz_file("DHS_peak.xyz")

        return None

    @property
    def macro_iter(self):
        """Total number of DHS steps taken so far"""
        return (
            len(self._initial_species_hist) + len(self._final_species_hist) - 2
        )

    def _step(self, side: str) -> None:
        """
        Take a DHS step, on the side requested, along the distance
        vector between the two images

        Args:
            side (str):
        """
        # todo coord system -- ditch internals?
        # take a DHS step by minimizing the distance by factor
        new_dist = (1 - self._reduction_fac) * self.imgpair.dist
        dist_vec = self.imgpair.dist_vec
        step = dist_vec * self._reduction_fac  # ??

        if side == "left":
            new_coord = self.imgpair.left_coord - step
            self.imgpair.left_coord = new_coord
        elif side == "right":
            new_coord = self.imgpair.right_coord + step
            self.imgpair.right_coord = new_coord
        else:
            raise ImgPairSideError()

        logger.info(
            f"DHS step on {side} image:" f" setting distance to {new_dist:.4f}"
        )
        assert self.imgpair.dist == new_dist

        return None

    def _has_jumped_over_barrier(self, coord: OptCoordinates, side: "str"):
        """
        Determines if the provided image after micro-iteration
        optimisation has jumped over the barrier towards the
        other species.
        """
        if side == "left":
            last_coord = self._initial_species_hist[-1]
            other_image = self._final_species_hist[-1]
        elif side == "right":
            last_coord = self._final_species_hist[-1]
            other_image = self._initial_species_hist[-1]
        else:
            raise ImgPairSideError()

        # DHS assumes the next point will lie between the
        # last two images on both side. If the current coord
        # is further from last coord on the same side than the
        # opposite image, then it has jumped over
        dist_to_last = np.linalg.norm(coord - last_coord)
        dist_before_step = np.linalg.norm(other_image - last_coord)

        if dist_to_last >= dist_before_step:
            return True
        else:
            return False

    def get_peak_species(self) -> autode.species.species.Species:
        """Obtain the highest energy point in the DHS energy path"""
        total_hist = (
            self._initial_species_hist + self._final_species_hist[::-1]
        )
        energies = [coord.e for coord in total_hist]
        assert all(energy is not None for energy in energies)
        max_idx = np.argmax(energies)
        tmp_spc = self._species.new_species("peak")
        tmp_spc.coordinates = total_hist[max_idx]
        tmp_spc.energy = total_hist[max_idx].e
        return tmp_spc

    def write_trajectories(self) -> None:
        """
        Writes .xyz trajectories for the DHS procedure;
        "initial_species_dhs.trj.xyz" for the initial_species
        supplied, "final_species_dhs.trj.xyz" for the final_species
        supplied, and "total_dhs.trj.xyz" which is a concatenated
        version with the trajectory of the final species reversed
        (i.e. the MEP calculated by DHS)
        """
        init_trj = "initial_species_dhs.trj.xyz"
        fin_trj = "final_species_dhs.trj.xyz"
        total_trj = "total_dhs.trj.xyz"
        if self.macro_iter < 2:
            logger.warning("Cannot write trajectory, not enough DHS points")
            return None

        if _rename_if_old_file_present(init_trj):
            self._write_trj_from_history(init_trj, self._initial_species_hist)
        else:
            logger.error(
                f"File: {init_trj} already exists, cannot write trajectory"
            )

        if _rename_if_old_file_present(fin_trj):
            self._write_trj_from_history(fin_trj, self._final_species_hist)
        else:
            logger.error(
                f"File: {fin_trj} already exists, cannot write trajectory"
            )

        if _rename_if_old_file_present(total_trj):
            total_hist = (
                self._initial_species_hist + self._final_species_hist[::-1]
            )
            self._write_trj_from_history(total_trj, total_hist)
        else:
            logger.error(
                f"File: {total_trj} already exists, cannot write trajectory"
            )

        return None

    def _write_trj_from_history(
        self, filename: str, hist: _OptimiserHistory
    ) -> None:
        """Convenience function to write trajectory from coord history"""
        tmp_spc = self._species.copy()

        for coord in hist:
            tmp_spc.coordinates = coord
            atoms_to_xyz_file(
                atoms=tmp_spc.atoms,
                filename=filename,
                title_line=f"DHS image E={coord.e:.6f} Ha",
                append=True,
            )

        return None

    def plot_energies(self):
        """
        Plot the energies along the MEP path as obtained by the DHS
        procedure. The points arising from the initial_species are
        shown in blue, which those from final species are shown in green.
        The x-axis shows euclidean distance of each MEP point from
        the initial_species geometry supplied, and the y-axis shows
        the energies in kcal/mol
        """
        if self.macro_iter < 2:
            logger.warning("Cannot plot energies, not enough points")
            return None

        plot_name = "DHS_MEP_path.pdf"

        if not _rename_if_old_file_present(plot_name):
            logger.error(
                f"File: {plot_name} already exists, "
                f"cannot write energy plot"
            )
            return None

        distances_init = []
        distances_fin = []
        energies_init = []
        energies_fin = []
        # starting coordinates
        init_coord = self._initial_species_hist[0]
        lowest_en = min(
            coord.e.to("kcalmol-1")
            for coord in (
                self._initial_species_hist + self._final_species_hist
            )
        )

        def put_dist_en_in_lists(dist_list, en_list, history):
            for coord in history:
                dist = float(np.linalg.norm(coord - init_coord))
                dist_list.append(dist)
                assert coord.e is not None
                en_list.append(float(coord.e.to("kcalmol-1")) - lowest_en)

        put_dist_en_in_lists(
            distances_init, energies_init, self._initial_species_hist
        )
        put_dist_en_in_lists(
            distances_fin, energies_fin, self._final_species_hist
        )

        # get the pyplot axes
        fig, ax = plt.subplots()
        if len(distances_init) > 0:
            ax.plot(distances_init, energies_init, "bo-")
        if len(distances_fin) > 0:
            ax.plot(distances_fin, energies_fin, "go-")
        ax.set_xlabel(f"Distance from initial image ({angstrom.plot_name})")
        ax.set_ylabel("Relative electronic energy (kcal/mol)")
        # todo beautify
        dpi = 400 if Config.high_quality_plots else 200
        fig.savefig(plot_name, dpi=dpi, bbox_inches="tight")


def _rename_if_old_file_present(filename: str) -> bool:
    """
    Renames the file if it is present in the current
    directory

    Args:
        filename:

    Returns:
        (bool): True if succeeded in renaming (or did not exist),
                False otherwise
    """
    rename_success = False
    if os.path.isfile(filename):
        for i in range(100):
            new_filename = filename + f"_old_{i}"
            if os.path.isfile(new_filename):
                continue
            os.rename(filename, new_filename)
            rename_success = True
            break
        if not rename_success:
            return False
    return True
