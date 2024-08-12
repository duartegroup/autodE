"""Partitioned rational function optimisation"""
import numpy as np
from typing import Union, Optional

from autode.log import logger
from autode.values import Distance
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.opt.coordinates.cartesian import CartesianCoordinates


class PRFOptimiser(CRFOptimiser):
    def __init__(
        self,
        init_alpha: Union[Distance, float] = 0.05,
        recalc_hessian_every: int = 10,
        imag_mode_idx: int = 0,
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

            imag_mode_idx: Index of the imaginary mode to follow. Default
                        is 0th mode i.e. the most negative mode

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = Distance(init_alpha, units="ang")
        assert self.alpha > 0
        self.recalc_hessian_every = int(recalc_hessian_every)
        self._mode_idx = imag_mode_idx
        self._last_eigvec: Optional[np.ndarray] = None  # store last mode
        self._hessian_update_types = [BofillUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""
        assert self._coords is not None and self._coords.g is not None

        if self.should_calculate_hessian:
            self._update_hessian()
        elif self.iteration != 0:
            self._coords.update_h_from_old_h(
                self._history.penultimate, self._hessian_update_types
            )

        assert self._coords.h is not None  # must set .h

        b, u = np.linalg.eigh(self._coords.h)
        f = u.T.dot(self._coords.g)
        n_negative_eigenvalues = sum(lmda < 0 for lmda in b)
        logger.info(
            f"∇^2E has {n_negative_eigenvalues} negative "
            f"eigenvalue(s). Should have 1"
        )

        imag_idx = self._get_imag_mode_idx(u)
        logger.info(f"Following mode {imag_idx} uphill")

        b_max = b[imag_idx]
        u_max = u[:, imag_idx]
        f_max = f[imag_idx]

        b_min = np.delete(b, imag_idx)
        u_min = np.delete(u, imag_idx, axis=1)
        f_min = np.delete(f, imag_idx)

        n = len(b)
        delta_s = np.zeros(shape=(n,))
        # downhill step
        aug_h_min = np.zeros(shape=(n, n))
        aug_h_min[: n - 1, : n - 1] = np.diag(b_min)
        aug_h_min[:-1, -1] = aug_h_min[-1, :-1] = f_min
        lambda_n = np.linalg.eigvalsh(aug_h_min)[0]
        logger.info(f"Calculated λ_n = {lambda_n:.6f}")

        for i in range(n - 1):
            delta_s -= f_min[i] * u_min[:, i] / (b_min[i] - lambda_n)

        # uphill step
        aug_h_max = np.zeros(shape=(2, 2))
        aug_h_max[:1, :1] = b_max
        aug_h_max[:-1, -1] = aug_h_max[-1, :-1] = f_max
        lambda_p = np.linalg.eigvalsh(aug_h_max)[-1]
        logger.info(f"Calculated λ_p = {lambda_p:.6f}")

        delta_s -= f_max * u_max / (b_max - lambda_p)

        self._last_eigvec = u[:, imag_idx].flatten()
        self._take_step_within_trust_radius(delta_s)
        return None

    def _get_imag_mode_idx(self, u: np.ndarray):
        """
        Find the imaginary mode to follow upwards in the current step.

        Args:
            u (np.ndarray): The Hessian eigenvectors

        Returns:
            (int): Integer
        """
        if self.iteration == 0:
            return self._mode_idx

        overlaps = []
        for i in range(u.shape[1]):
            overlaps.append(
                np.abs(np.dot(u[:, i].flatten(), self._last_eigvec))
            )

        mode_idx = np.argmax(overlaps)
        logger.info(f"Overlap with previous TS mode: {overlaps[mode_idx]:.3f}")
        return mode_idx

    def _initialise_run(self) -> None:
        """
        Initialise running a partitioned rational function optimisation by
        setting the coordinates and Hessian
        """
        assert self._species is not None, "Must have a species to init"

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
