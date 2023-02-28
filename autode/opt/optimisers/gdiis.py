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
        use_last_hessian=True,
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
        self._space = history[: self._max_vec]

        if use_last_hessian:
            self._H = history[-1].h
        else:
            self._H = None

        self._check_stable = bool(check_stability)
        self._trust = trust

    def calculate(self):
        n = len(self._space)

        if self._H is None:
            self._H = np.eye(n)

        orig_err_vecs = [-np.matmul(self._H, coord.g) for coord in self._space]
        orig_err_vecs = orig_err_vecs[::-1]  # reverse to get the latest first

        # rescale error vectors by dividing by the smallest error vector norm
        # criteria (d), page 13, Phys. Chem. Chem. Phys., 2002, 4, 11–15
        scale_fac = min(
            [np.linalg.norm(err_vec) for err_vec in orig_err_vecs]
        )
        err_vecs = [err_vec / scale_fac for err_vec in orig_err_vecs]

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
            # check for singularity of A matrix nonetheless
            coeff_size = np.linalg.norm(coeffs)
            if coeff_size >= 1.0e8:
                break
            # eq (9) coeffs = c/|r|^2, so sum(coeffs) = 1/|r|^2 as sum(c) = 1
            scaled_coeffs = coeffs / np.sum(coeffs)
            if not self._check_stable:
                gdiis_coeffs = scaled_coeffs
                continue

            is_angle_valid = self._check_against_ref_step()





