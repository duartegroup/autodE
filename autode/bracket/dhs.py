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
    def __init__(
        self,
        fn: Callable,
        x0: np.ndarray,
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
        self._iter = 1

        self._x = None
        self._last_x = None
        self._en = None
        self._last_en = None
        self._grad = None
        self._last_grad = None
        self._hess = None
        self._pred_de = None

    def run(self):
        self._x = self._x0
        dim = self._x.shape[0]  # dimension of problem

        for i in range(self._maxiter):
            self._last_en, self._last_grad = self._en, self._grad
            self._en, self._grad = self._fn(self._x)
            self._hess = self._get_hessian()
            assert self._grad.shape[0] == dim
            self._qnr_restricted_step()

    def _get_hessian(self):
        if self._iter == 1:
            return np.eye(self._x.shape[0])
        # todo update hessian here
        # todo make hessian positive definite

    def _qnr_restricted_step(self):
        self._update_trust_radius()
        grad = self._grad.reshape(-1, 1)
        inv_hess = np.linalg.inv(self._hess)
        step = (inv_hess @ grad)
        step_size = np.linalg.norm(step)
        if step_size > self._trust:
            step = step * (self._trust / step_size)

        self._pred_de = grad.T @ step + 0.5 * (step.T @ self._hess @ step)
        self._last_x = self._x.copy()
        self._x = self._x + step.flatten()

    def _update_trust_radius(self):
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
                method='CG',
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


class DHS_old:
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        maxiter: int = 10,
        reduction_factor: float = 0.05,
        dist_tol: float = 0.05,
        init_trust: Distance = Distance(-0.08, "ang"),
        max_trust: Distance = Distance(0.2, "ang"),
    ):
        self.imgpair = DistanceConstrainedImagePair(
            initial_species, final_species
        )
        self._reduction_fac = reduction_factor

        if int(maxiter) <= 0:
            raise ValueError(
                "An optimiser must be able to run at least one "
                f"step, but tried to set maxiter = {maxiter}"
            )

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, units="ang")
        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None

        self._grad = None
        self._hess = None

        self._left_trust = init_trust.to("ang")
        self._right_trust = init_trust.to("ang")
        if init_trust < 0:
            self._trust_update = False
        else:
            self._trust_update = True
        self._max_trust = max_trust.to("ang")
        self._left_pred_delta_e = None
        self._right_pred_delta_e = None

        # todo have a history where store the optimised points after each macroiter

    @property
    def macroiter_converged(self):
        if self.imgpair.euclid_dist <= self._dist_tol:
            return True
        else:
            return False

    @property
    def microiter_converged(self):
        # gtol and dist_tol
        pass

    @property
    def iteration(self) -> int:
        return self.imgpair.total_iters

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        return True if self.imgpair.total_iters > self._maxiter else False

    def calculate(
        self,
        engrad_method: _optional_method = None,
        hess_method: _optional_method = None,
        n_cores: Optional[int] = None,
    ):
        from autode.methods import (
            method_or_default_hmethod,
            method_or_default_lmethod,
        )

        engrad_method = method_or_default_hmethod(engrad_method)
        hess_method = method_or_default_lmethod(hess_method)
        if n_cores is None:
            n_cores = Config.n_cores
        self.imgpair.set_method_and_n_cores(engrad_method=engrad_method, n_cores=n_cores, hess_method=hess_method)

        self._initialise_run()

        # in each macroiter, the distance criteria is reduced by factor
        macroiter_num = 0
        maxiter_reached = False
        while not self.macroiter_converged:
            self.imgpair.target_dist = self.imgpair.euclid_dist * (
                1 - self._reduction_fac
            )
            macroiter_num += 1
            while not self.microiter_converged:
                if self.imgpair.left_coord.e < self.imgpair.right_coord.e:
                    side = "left"
                else:
                    side = "right"
                self._take_one_sided_step(side)
                self._update_one_side_mol_engrad_hess(side)
                self._update_trust_radius(side)

                if self._exceeded_maximum_iteration:
                    # error message
                    maxiter_reached = True
                    break
            if maxiter_reached:
                break

        pass

    def _take_one_sided_step(self, side: str):
        grad = self.imgpair.get_one_img_lagrangian_gradient(side)
        hess = self.imgpair.get_one_img_lagrangian_hessian(side)
        self._prfo_step(side, grad, hess)

    def _update_one_side_mol_engrad_hess(self, side: str):
        self.imgpair.update_one_img_molecular_engrad(side)
        self.imgpair.update_one_img_molecular_hessian_by_formula(side)

    def _initialise_run(self):
        self.imgpair.update_one_img_molecular_engrad("left")
        self.imgpair.update_one_img_molecular_hessian_by_calc("left")
        self.imgpair.update_one_img_molecular_engrad("right")
        self.imgpair.update_one_img_molecular_hessian_by_calc("right")
        return None

    def _prfo_step(self, side: str, grad: np.ndarray, hess: np.ndarray):
        b, u = np.linalg.eigh(hess)
        f = u.T.dot(grad)

        delta_s = np.zeros(shape=(self.imgpair.n_atoms,))

        # partition the RFO
        # The constraint is the last item
        constr_components = u[-1, :]
        # highest component of constraint must be the required mode
        constr_mode = np.argmax(constr_components)
        u_max = u[:, constr_mode]
        b_max = b[constr_mode]
        f_max = f[constr_mode]

        # only one constraint - distance
        aug_h_max = np.zeros(shape=(2, 2))
        aug_h_max[0, 0] = b_max
        aug_h_max[1, 0] = f_max
        aug_h_max[0, 1] = f_max
        lambda_p = np.linalg.eigvalsh(aug_h_max)[-1]

        delta_s -= f_max * u_max / (b_max - lambda_p)

        u_min = np.delete(u, constr_mode, axis=1)
        b_min = np.delete(b, constr_mode)
        f_min = np.delete(f, constr_mode)
        # todo deal with rot./trans.?

        # n_atoms non-constraint modes
        m = self.imgpair.n_atoms
        aug_h_min = np.zeros(shape=(m + 1, m + 1))
        aug_h_min[:m, :m] = np.diag(b_min)
        aug_h_min[-1, :m] = f_min
        aug_h_min[:m, -1] = f_min
        eigvals_min = np.linalg.eigvalsh(aug_h_min)
        min_mode = np.where(np.abs(eigvals_min) > 1e-15)[0][0]
        lambda_n = eigvals_min[min_mode]

        for i in range(m):
            delta_s -= f_min[i] * u_min[:, i] / (b_min[i] - lambda_n)

        self._take_step_within_trust_radius(side, delta_s, grad, hess)

    def _take_step_within_trust_radius(
        self,
        side: str,
        delta_s: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
    ):
        # set the coords to one side
        # todo fix predicted e change
        delta_s = delta_s.flatten()
        step_length = np.linalg.norm(delta_s)

        trust = self._left_trust if side == "left" else self._right_trust

        if step_length > trust:
            factor = float(trust / step_length)
            delta_s = delta_s * factor

        pred_delta_e = float(np.dot(grad, delta_s))
        pred_delta_e += 0.5 * float(
            delta_s.reshape(1, -1) @ hess @ delta_s.reshape(-1, 1)
        )

        if side == "left":
            new_coord = self.imgpair.left_coord + delta_s[:-1]
            self.imgpair.left_coord = new_coord
            # todo need to update lagrangian multiplier
            self._left_pred_delta_e = pred_delta_e
        elif side == "right":
            new_coord = self.imgpair.right_coord + delta_s
            self.imgpair.right_coord = new_coord
            self._right_pred_delta_e = pred_delta_e
        else:
            raise Exception

        return None

    def _update_trust_radius(self, side: str) -> None:
        if not self._trust_update:
            return None
        # todo should this consider lagrangian or just the molecule
        _, _, hist, _ = self.imgpair._get_img_by_side()

        pass
