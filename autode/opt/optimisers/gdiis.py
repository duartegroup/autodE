"""
Geometry Optimisation by Direction Interpolation in
Iterative Subspace (GDIIS)

Geometry Optimisation by Energy-represented DIIS (GEDIIS)

"""
import numpy as np

from autode.opt.optimisers.base import _OptimiserHistory
from autode.opt.coordinates import OptCoordinates


class GDIIS:
    """Stabilised GDIIS"""

    def __init__(
        self,
        history: _OptimiserHistory,
        max_vecs=5,
        use_hessian=True,
        check_stability=True,
        trust=None,
    ):

        if not isinstance(history, list):
            raise TypeError(
                "History must be a list of coordinates or " "_OptimiserHistory"
            )

        if not isinstance(history[0], OptCoordinates):
            raise TypeError("Supplied history must contain OptCoordinates")

        if len(history) < 2:
            raise ValueError("History must have at least two coordinates")

        # todo warn if max_vec larger than history
        self._max_vec = int(max_vecs)
        self._space = history[::-1][:self._max_vec]  # reverse to get the latest first

        if use_hessian:
            self._H, self._ref_step = self.get_qnr_step()
        else:
            self._H, self._ref_step = None, None

        self._check_stable = bool(check_stability)
        self._trust = trust

    def calculate(self):
        n = len(self._space)

        if self._H is None:
            self._H = np.eye(len(self._space[0].g))

        err_vecs = [-np.matmul(self._H, coord.g) for coord in self._space]

        # rescale error vectors by dividing by the smallest error vector norm
        # criteria (d), page 13, Phys. Chem. Chem. Phys., 2002, 4, 11–15
        scale_fac = min(
            [np.linalg.norm(err_vec) for err_vec in err_vecs]
        )
        err_vecs = [err_vec / scale_fac for err_vec in err_vecs]

        gdiis_coeffs = None
        for num_vec in range(2, min(self._max_vec, n) + 1):
            print(f"Attempting GDIIS with {num_vec} previous geometries")

            # eq (9) in Phys. Chem. Chem. Phys., 2002, 4, 11–15
            a_mat = np.zeros(shape=(num_vec, num_vec))
            for i in range(num_vec):
                for j in range(num_vec):
                    a_mat[i, j] = np.dot(err_vecs[i], err_vecs[j])
            try:
                coeffs = np.linalg.solve(a_mat, np.ones(num_vec))
            except np.linalg.LinAlgError:
                break
            # check for singularity of A matrix always
            coeff_size = np.linalg.norm(coeffs)
            if coeff_size >= 1.0e8:
                break
            # eq (9) coeffs = c/|r|^2, so sum(coeffs) = 1/|r|^2 as sum(c) = 1
            scaled_coeffs = coeffs / np.sum(coeffs)
            if not self._check_stable:
                gdiis_coeffs = scaled_coeffs
                continue
            pred_coord = self._get_gdiis_coord(scaled_coeffs)
            is_angle_valid = self._check_against_ref_step()

    def _get_gdiis_coord(self, coeffs):
        """Get predicted coordinate for GDIIS"""
        # todo do we need original error vectors or new vectors
        pass

    def _get_qnr_step(self):
        # try RFO step first
        hess = self._space[-1].h
        grad = self._space[-1].g
        assert hess is not None
        h_n = hess.shape[0]

        aug_h = np.zeros(shape=(h_n+1, h_n+1))

        aug_h[:h_n, :h_n] = hess
        aug_h[-1, :h_n] = grad
        aug_h[:h_n, -1] = grad

        aug_h_lmda, aug_h_v = np.linalg.eigh(aug_h)
        mode = np.where(np.abs(aug_h_lmda) > 1.e-15)[0][0]

        step = aug_h_v[:-1, mode] / aug_h_v[-1, mode]

        # todo if step is not within trust radius then do
        # qa
        h_eff = hess - aug_h_lmda[mode] * np.eye(h_n)
        # todo check this formula
        return h_eff, step.flatten()

    def _check_against_ref_step(self, gdiis_step):
        pass





