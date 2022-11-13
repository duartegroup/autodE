"""Partitioned rational function optimisation"""
import numpy as np
from autode.log import logger
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.opt.optimisers.hessian_update import BofillUpdate, NullUpdate
from autode.exceptions import CalculationException


class PRFOptimiser(CRFOptimiser):
    def __init__(self, init_alpha: float = 0.05, *args, **kwargs):
        """
        Partitioned rational function optimiser (PRFO) using a maximum step
        size of alpha trying to maximise along a mode while minimising along
        all others to locate a transition state (TS)

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size

            imag_mode_idx: Index of the imaginary mode to follow. Default = 0,
                           the most imaginary mode (i.e. most negative
                           eigenvalue)

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = float(init_alpha)
        self._hessian_update_types = [BofillUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""

        self._coords.h = self._updated_h()

        idxs = list(range(len(self._coords)))

        b, u = np.linalg.eigh(self._coords.h[:, idxs][idxs, :])
        logger.info(
            f"∇^2L has {sum(lmda < 0 for lmda in b)} negative "
            f"eigenvalue(s). Should have 1"
        )

        logger.info("Calculating transformed gradient vector")
        f = u.T.dot(self._coords.g[idxs])

        lambda_p = self._lambda_p_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_p=+{lambda_p:.8f}")

        lambda_n = self._lambda_n_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_n={lambda_n:.8f}")

        delta_s = np.zeros(shape=(len(idxs),))
        delta_s -= f[0] * u[:, 0] / (b[0] - lambda_p)

        for j in range(1, len(idxs)):
            delta_s -= f[j] * u[:, j] / (b[j] - lambda_n)

        _ = self._take_step_within_trust_radius(delta_s)
        return None

    def _initialise_run(self) -> None:
        """
        Initialise running a partitioned rational function optimisation by
        setting the coordinates and Hessian
        """
        self._coords = CartesianCoordinates(self._species.coordinates).to(
            "dic"
        )
        self._update_hessian_gradient_and_energy()
        return None
