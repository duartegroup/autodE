from typing import Optional, List, Tuple
import numpy as np

from autode.values import Distance, PotentialEnergy, Gradient
from autode.bracket.imagepair import DistanceConstrainedImagePair

from autode.utils import work_in_tmp_dir, ProcessPool
from autode.log import logger
from autode.config import Config

import autode.species.species
import autode.wrappers.methods


_optional_method = Optional[autode.wrappers.methods.Method]


class DHS:
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        maxiter: int = 10,
        reduction_factor: float = 0.05,
        dist_tol: float = 0.05,
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
        self.imgpair.set_method_and_n_cores(
            engrad_method=engrad_method,
            hess_method=hess_method,
            n_cores=n_cores,
        )

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
                # in gradient update step, update hessian as well
                # no need to update hessian in prfo step
                if self._exceeded_maximum_iteration:
                    # error message
                    maxiter_reached = True
                    break
            if maxiter_reached:
                break

        pass

    def _take_one_sided_step(self, side: str):
        grad = self.imgpair.get_one_img_lagrangian_gradient(side)
        hess = self.imgpair.get_one_sided_lagrangian_hessian(side)
        self._prfo_step(side, grad, hess)

    def _update_one_side_mol_engrad_hess(self, side: str):
        self.imgpair.update_one_img_molecular_engrad(side)
        # todo interpolate hessian by Bofill
        # self.imgpair.update_one_side_molecular_hessian(side)

    def _initialise_run(self):
        # todo this func is empty
        # todo is it really necessary to parallelise in DHS?
        self.imgpair.update_one_img_molecular_engrad("left")
        self.imgpair.update_one_img_molecular_hessian_by_calc("left")
        self.imgpair.update_one_img_molecular_engrad("right")
        self.imgpair.update_one_img_molecular_hessian_by_calc("right")
        pass

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

        self.take_step_within_trust_radius(side, delta_s)

    def take_step_within_trust_radius(self, side, delta_s):
        # set the coords to one side
        pass
