import os
from typing import Optional, Callable, Tuple
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from matplotlib import pyplot as plt

from autode.values import Distance, PotentialEnergy, Gradient, Energy
from autode.bracket.imagepair import BaseImagePair
from autode.opt.optimisers.base import _OptimiserHistory
from autode.opt.optimisers.hessian_update import BFGSPDUpdate, NullUpdate
from autode.opt.coordinates.base import _ensure_positive_definite
from autode.input_output import atoms_to_xyz_file

from autode.utils import work_in_tmp_dir, ProcessPool
from autode.log import logger
from autode.config import Config

import autode.species.species
import autode.wrappers.methods


class DHSImagePair(BaseImagePair):
    """
    An image-pair that defined the distance between two
    images as the Euclidean distance (square of root of
    sum of the squares of deviations in Cartesian)
    """
    @property
    def dist_vec(self) -> np.ndarray:
        """The distance vector pointing to right_image from left_image"""
        return np.array(self.left_coord - self.right_coord)

    def get_one_img_perp_grad(self, side: str):
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
        perp_grad = coord.g - proj_grad

        return perp_grad

    @property
    def euclid_dist(self):
        """The Euclidean distance between the images"""
        return Distance(np.linalg.norm(self.dist_vec), 'Ang')


def _set_one_img_coord_and_get_engrad(
        coord: np.array,
        side: str,
        imgpair: DHSImagePair) -> Tuple[float, np.ndarray]:
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

    """
    if side == 'left':
        imgpair.left_coord = np.array(coord).flatten()
    elif side == 'right':
        imgpair.right_coord = np.array(coord).flatten()
    else:
        raise Exception

    imgpair.update_one_img_molecular_engrad(side)
    new_coord = imgpair.get_coord_by_side(side)
    en = float(new_coord.e.to('Ha'))
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
        dist_tol: Distance = Distance(0.6, 'ang'),
    ):
        """
        Dewar-Healy-Stewart method to find transition state.
        1) The order of initial_species/final_species
        does not matter and can be interchanged; 2) The
        reduction_factor is 0.05 or 5% by default, which is
        quite conservative, so may want to increase that
        if convergence is slow; 3) The distance tolerance
        should not be lowered any more than 0.5 Angstrom
        as DHS is unstable when the distance is low, and has
        a tendency for one image to jump over the barrier

        Args:
            initial_species: The "reactant" species
            final_species: The "product" species
            maxiter: Maximum number of en/grad evaluations
            reduction_factor: The factor by which the distance is
                              decreased in each DHS step
            dist_tol: The distance tolerance at which DHS will
                      stop, values <0.5 Angstrom are not recommended!
        """
        self.imgpair = DHSImagePair(initial_species, final_species)
        self._species = initial_species.copy()  # just hold the species
        self._reduction_fac = float(reduction_factor)

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, 'ang')

        # these only hold the coords after finishing optimisation
        # for each DHS step. Put the initial coordinates here
        self._initial_species_hist = _OptimiserHistory()
        self._initial_species_hist.append(self.imgpair.left_coord)
        self._final_species_hist = _OptimiserHistory()
        self._final_species_hist.append(self.imgpair.right_coord)

    @property
    def converged(self):
        """Is DHS converged to the desired distance tolerance?"""
        if self.imgpair.euclid_dist < self._dist_tol:
            return True
        else:
            return False

    def calculate(self, method, n_cores):
        """
        Run the DHS calculation. Can only be called once!

        Args:
            method:
            n_cores:

        Returns:

        """
        self.imgpair.set_method_and_n_cores(method, n_cores)
        self.imgpair.update_one_img_molecular_engrad('left')
        self.imgpair.update_one_img_molecular_engrad('right')
        logger.error("Starting DHS optimisation")
        while not self.converged:
            if self.imgpair.left_coord.e < self.imgpair.right_coord.e:
                side = 'left'
                hist = self._initial_species_hist
            else:
                side = 'right'
                hist = self._final_species_hist
            self._step(side)
            coord0 = self.imgpair.get_coord_by_side(side)
            curr_maxiter = self._maxiter - self.imgpair.total_iters
            # scipy does micro-iterations
            res = scipy_minimize(
                fun=_set_one_img_coord_and_get_engrad,
                jac=True,
                x0=coord0,
                args=(side, self.imgpair),
                method='CG',
                options={'gtol': 0.001, 'disp': True, 'maxiter': curr_maxiter}
            )  # is conjugate gradients good enough?
            new_coord = self.imgpair.get_coord_by_side(side)
            rms_grad = np.sqrt(np.mean(np.square(new_coord.g)))
            if res['success']:
                logger.error("Successful optimization after DHS step, final"
                             f" RMS gradient = {rms_grad} Ha/angstrom")
            else:
                break
            hist.append(new_coord.copy())
            self._log_convergence()

    @property
    def macro_iter(self):
        """Total number of DHS steps taken so far"""
        return (
            len(self._initial_species_hist)
            + len(self._final_species_hist)
            - 2
        )

    def _step(self, side: str):
        # take a DHS step by minimizing the distance by factor
        new_dist = (1 - self._reduction_fac) * self.imgpair.euclid_dist
        dist_vec = self.imgpair.dist_vec
        step = dist_vec * self._reduction_fac  # ??

        if side == 'left':
            new_coord = self.imgpair.left_coord - step
            self.imgpair.left_coord = new_coord
        elif side == 'right':
            new_coord = self.imgpair.right_coord + step
            self.imgpair.right_coord = new_coord

        logger.error(f"DHS step: target distance = {new_dist}")

        assert self.imgpair.euclid_dist == new_dist

    def _log_convergence(self):
        logger.error(
            f"Macro-iteration #{self.macro_iter}: Distance = "
            f"{self.imgpair.euclid_dist:.4f}; Energy (initial species) = "
            f"{self.imgpair.left_coord.e}; Energy (final species) = "
            f"{self.imgpair.right_coord.e}"
        )

    def write_trajectories(self):

        if os.path.isfile('initial_species.trj.xyz'):
            logger.error("File: initial_species.trj.xyz already "
                         "exists, cannot write trajectory")
        else:
            self._write_trj_from_history(
                'initial_species.trj.xyz',
                self._initial_species_hist
            )

        if os.path.isfile('final_species.trj.xyz'):
            logger.error("File: final_species.trj.xyz already "
                         "exists, cannot write trajectory")
        else:
            self._write_trj_from_history(
                'final_species.trj.xyz',
                self._final_species_hist
            )

        if os.path.isfile('total_dhs.trj.xyz'):
            logger.error("File: total_dhs.trj.xyz already "
                         "exists, cannot write trajectory")
        else:
            total_hist = (self._initial_species_hist
                          + self._final_species_hist[::-1])
            self._write_trj_from_history(
                'total_dhs.trj.xyz',
                total_hist
            )

    def _write_trj_from_history(self,
                                filename: str,
                                hist: _OptimiserHistory):
        tmp_spc = self._species.copy()

        for coord in hist:
            tmp_spc.coordinates = coord
            atoms_to_xyz_file(
                atoms=tmp_spc.atoms,
                filename=filename,
                title_line=f"DHS image E={coord.e:.6f} Ha",
                append=True
            )

    def plot_energies(self):
        plot_name = "DHS_MEP_path.pdf"
        if os.path.isfile(plot_name):
            logger.error(f"File: {plot_name} already exists, "
                         f"cannot write energy plot")
            return

        total_hist = (self._initial_species_hist
                      + self._final_species_hist[::-1])
        # starting coordinate of the reactant
        initial_coord = total_hist[-1]
        distances = []
        energies = []
        for coord in total_hist:
            dist = float(np.linalg.norm(coord - initial_coord))
            en = float(coord.e.to('Ha'))
            distances.append(dist)
            energies.append(en)

        # get the pyplot axes
        fig, ax = plt.subplots()
        ax.plot(distances, energies)
        # todo beautify
        dpi = 400 if Config.high_quality_plots else 200
        fig.savefig(plot_name, dpi=dpi, bbox_inches='tight')

