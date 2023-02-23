"""
Dewar-Healy-Stewart Method for finding transition states

As described in J. Chem. Soc. Farady Trans. 2, 1984, 80, 227-233
"""

import os
from typing import Tuple, Union
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from autode.values import Distance
from autode.units import ang as angstrom
from autode.bracket.imagepair import BaseImagePair, ImgPairSideError
from autode.methods import get_lmethod
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.optimisers.base import _OptimiserHistory
from autode.input_output import atoms_to_xyz_file

from autode.log import logger
from autode.config import Config

import autode.species.species
import autode.wrappers.methods


class DHSImagePair(BaseImagePair):
    """
    An image-pair that defines the distance between two images
    as the Euclidean distance (square of root of sum of the
    squares of deviations in Cartesian coordinates)
    """

    @property
    def dist_vec(self) -> np.ndarray:
        """The distance vector pointing to right_image from left_image"""
        return np.array(self.left_coord - self.right_coord)

    @property
    def euclid_dist(self) -> Distance:
        """The Euclidean distance between the images"""
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
        _, coord, _, _ = self._get_img_by_side(side)
        # project the gradient towards the distance vector
        proj_grad = dist_vec * coord.g.dot(dist_vec) / dist_vec.dot(dist_vec)
        # gradient component perpendicular to distance vector
        perp_grad = np.array(coord.g - proj_grad).flatten()

        return perp_grad


def _set_one_img_coord_and_get_engrad(
    coord: np.array,
    side: str,
    imgpair: DHSImagePair,
) -> Tuple[float, np.ndarray]:
    """
    Convenience function that allows setting coordinates
    and obtaining energy and gradient at the same time.
    To be called by the scipy minimizer (or any other
    minimizer)

    Args:
        coord: The coordinates in an array
        side: The side of imagepair that is updated
        imgpair: The imagepair object (DHS)

    Returns:
        (tuple[float, ndarray]): energy, gradient in flat array
    """
    if side == "left":
        imgpair.left_coord = np.array(coord).flatten()
    elif side == "right":
        imgpair.right_coord = np.array(coord).flatten()
    else:
        raise ImgPairSideError()

    imgpair.update_one_img_molecular_engrad(side)
    new_coord = imgpair.get_coord_by_side(side)
    en = float(new_coord.e.to("Ha"))
    grad = imgpair.get_one_img_perp_grad(side)
    return en, grad


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
        optimiser: str = "BFGS",
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

            optimiser: The optimiser to use for minimising after
                       the DHS step, choose scipy's 'CG' (conjugate
                       gradients) or 'BFGS' (or maybe 'L-BFGS-B')
        """
        self.imgpair = DHSImagePair(initial_species, final_species)
        self._species = initial_species.copy()  # just hold the species
        self._reduction_fac = abs(float(reduction_factor))
        assert self._reduction_fac < 1.0

        self._maxiter = abs(int(maxiter))
        self._dist_tol = Distance(dist_tol, "ang")
        self._opt_driver = str(optimiser)

        # these only hold the coords after finishing optimisation
        # for each DHS step. Put the initial coordinates here
        self._initial_species_hist = _OptimiserHistory()
        self._initial_species_hist.append(self.imgpair.left_coord)
        self._final_species_hist = _OptimiserHistory()
        self._final_species_hist.append(self.imgpair.right_coord)

    @property
    def converged(self) -> bool:
        """Is DHS converged to the desired distance tolerance?"""
        if self.imgpair.euclid_dist < self._dist_tol:
            return True
        else:
            return False

    def _initialise_run(self) -> None:
        """
        Initialise everything needed for the first DHS macro-iteration
        (Only energies are needed)
        """
        self.imgpair.update_one_img_molecular_energy("left")
        self.imgpair.update_one_img_molecular_energy("right")
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
                hist = self._initial_species_hist
            else:
                side = "right"
                hist = self._final_species_hist

            # take a step on the side with lower energy
            self._step(side)
            coord0 = np.array(self.imgpair.get_coord_by_side(side))

            # calculate the number of remaining maxiter to feed into scipy
            curr_maxiter = self._maxiter - self.imgpair.total_iters
            if curr_maxiter == 0:
                break
            # scipy does micro-iterations
            res = minimize(
                fun=_set_one_img_coord_and_get_engrad,
                x0=coord0,
                jac=True,
                args=(side, self.imgpair),
                method=self._opt_driver,
                options={"gtol": 5.0e-4, "maxiter": curr_maxiter},
            )

            # parse the scipy results and log output
            perp_grad = self.imgpair.get_one_img_perp_grad(side)
            rms_grad = np.sqrt(np.mean(np.square(perp_grad)))

            if res["success"]:
                logger.info(
                    "Successful optimization after DHS step, final"
                    f" RMS of projected gradient = {rms_grad:.6f}"
                    f" Ha/angstrom"
                )
            else:
                # step may be accepted if rms gradient is low enough
                if rms_grad < 1.0e-3:
                    logger.info(
                        "Optimization not converged completely,"
                        " but accepted on the basis of RMS"
                        f" gradient = {rms_grad:.6f} Ha/angstrom"
                        f" being low enough"
                    )
                else:
                    logger.error(
                        "Micro-iterations (optimisation) after a"
                        " DHS step did not converge, exiting"
                    )
                    break

            # if scipy succeeded
            new_coord = self.imgpair.get_coord_by_side(side)
            # check if it has jumped over the barrier
            if self._has_jumped_over_barrier(new_coord, side):
                logger.warning(
                    "One image has jumped over the other image"
                    " while running DHS optimisation. This"
                    " indicates that the distance between images"
                    " is quite close, so DHS cannot proceed even"
                    " though the distance criteria is not met"
                )
                break

            else:
                # otherwise put the coordinate into appropriate history
                hist.append(new_coord.copy())

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
            f"{self.imgpair.euclid_dist:.4f}; Energy (initial species) = "
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
        # take a DHS step by minimizing the distance by factor
        new_dist = (1 - self._reduction_fac) * self.imgpair.euclid_dist
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
        assert self.imgpair.euclid_dist == new_dist

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
