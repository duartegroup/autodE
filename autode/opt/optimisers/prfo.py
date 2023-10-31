"""Partitioned rational function optimisation"""
import numpy as np
from typing import Union

from autode.log import logger
from autode.values import Distance
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.exceptions import CalculationException


class PRFOptimiser(CRFOptimiser):
    def __init__(
        self,
        init_alpha: Union[Distance, float] = 0.05,
        recalc_hessian_every: int = 10,
        *args,
        **kwargs,
    ):
        """
        Partitioned rational function optimiser (PRFO) using a maximum step
        size of alpha trying to maximise along a mode while minimising along
        all others to locate a transition state (TS)

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size (default Å if unit not given)

            imag_mode_idx: Index of the imaginary mode to follow. Default = 0,
                           the most imaginary mode (i.e. most negative
                           eigenvalue)

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = Distance(init_alpha, units="ang")
        assert self.alpha > 0
        self.recalc_hessian_every = int(recalc_hessian_every)
        self._hessian_update_types = [BofillUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""
        assert self._coords is not None and self._coords.g is not None

        if self.should_calculate_hessian:
            self._update_hessian()
        else:
            self._coords.h = self._updated_h()

        assert self._coords.h is not None  # _update_hessian must set .h
        idxs = list(range(len(self._coords)))

        b, u = np.linalg.eigh(self._coords.h[:, idxs][idxs, :])
        n_negative_eigenvalues = sum(lmda < 0 for lmda in b)
        logger.info(
            f"∇^2E has {n_negative_eigenvalues} negative "
            f"eigenvalue(s). Should have 1"
        )

        if n_negative_eigenvalues < 1:
            raise CalculationException("Lost imaginary (TS) mode")

        f = u.T.dot(self._coords.g[idxs])
        lambda_p = self._lambda_p_from_eigvals_and_gradient(b, f)
        lambda_n = self._lambda_n_from_eigvals_and_gradient(b, f)
        logger.info(f"Calculated λ_p=+{lambda_p:.8f}, λ_n={lambda_n:.8f}")

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
        assert self._species is not None, "Must have a species to init"

        # self._build_internal_coordinates()
        self._coords = CartesianCoordinates(self._species.coordinates).to(
            "dic"
        )
        self._update_hessian_gradient_and_energy()
        return None

    @property
    def should_calculate_hessian(self) -> bool:
        """Should an explicit Hessian calculation be performed?"""
        n = self.iteration
        return n > 1 and n % self.recalc_hessian_every == 0
