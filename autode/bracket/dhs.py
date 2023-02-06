from typing import Optional, Callable
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from autode.values import Distance, PotentialEnergy, Gradient
from autode.bracket.imagepair import DistanceConstrainedImagePair
from autode.bracket.imagepair import BaseImagePair
from autode.opt.optimisers.base import _OptimiserHistory

from autode.utils import work_in_tmp_dir, ProcessPool
from autode.log import logger
from autode.config import Config

import autode.species.species
import autode.wrappers.methods


_optional_method = Optional[autode.wrappers.methods.Method]


class ScaledQNROptimiser:
    """
    A simple scaled step Quasi-NR optimiser with dynamic
    trust radius. Needs gradient and function value (i.e. energy
    for QM purposes)
    """
    def __init__(
        self,
        fn: Callable,
        x0,
        maxiter: int,
        gtol: float = 1e-3,
        init_trust: float = 0.06,
        max_trust: float = 0.1
    ):
        self._fn = fn
        self._x0 = np.array(x0).flatten()
        self._gtol = gtol
        self._maxiter = int(maxiter)
        self._trust = float(init_trust)
        self._max_trust = float(max_trust)

        self.en = None
        self.grad = None
        self.last_grad = None
        self.hess = None
        self.last_hess = None

    def run(self):
        x = self._x0
        dim = len(x)  # dimension of problem
        en, grad = self._fn(x)

        for i in range(self._maxiter):
            pass


class DHSImagePair(BaseImagePair):
    def __init__(
        self,
        left_image: autode.species.Species,
        right_image: autode.species.Species,
    ):
        super().__init__(left_image, right_image)

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
        return Distance(np.linalg.norm(self.dist_vec), 'Ang')


def _set_one_img_coord_and_get_engrad(
        coord: np.array,
        side: str,
        imgpair: DHSImagePair):
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
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        maxiter: int = 100,
        reduction_factor: float = 0.05,
        dist_tol: Distance = Distance(0.5, 'ang'),
    ):
        self.imgpair = DHSImagePair(initial_species, final_species)
        self._reduction_fac = float(reduction_factor)

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, 'ang')

        # these only hold the coords after optimisation
        self._initial_species_hist = _OptimiserHistory()
        self._final_species_hist = _OptimiserHistory()

    @property
    def converged(self):
        if self.imgpair.euclid_dist < self._dist_tol:
            return True
        else:
            return False

    def calculate(self, method, n_cores):
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
            res = scipy_minimize(
                fun=_set_one_img_coord_and_get_engrad,
                jac=True,
                x0=coord0,
                args=(side, self.imgpair),
                method='l-bfgs-b',
                options={'gtol': 0.001, 'disp': True}
            )
            new_coord = self.imgpair.get_coord_by_side(side)
            rms_grad = np.sqrt(np.mean(np.square(new_coord.g)))
            if res['success']:
                logger.error("Successful optimization after DHS step, final"
                             f" RMS gradient = {rms_grad} Ha/angstrom")
            hist.append(new_coord.copy())
            self._log_convergence()

    @property
    def macro_iter(self):
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
