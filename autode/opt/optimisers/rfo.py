import numpy as np
from autode.log import logger
from autode.opt.optimisers.base import NDOptimiser
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.hessian_update import BFGSUpdate, NullUpdate


class RFOOptimiser(NDOptimiser):
    """Rational function optimisation in delocalised internal coordinates"""

    def __init__(self,
                 init_alpha: float = 0.1,
                 *args,
                 **kwargs):
        """
        Rational function optimiser (RFO) using a maximum step size of alpha

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Initial step size

        See Also:

        """
        super().__init__(*args, **kwargs)

        self.alpha = init_alpha
        self._hessian_update_types = [BFGSUpdate, NullUpdate]

    def _step(self) -> None:
        """RFO step"""
        self._coords.h_inv = self._updated_h_inv()

        h_n, _ = self._coords.h.shape

        # Form the augmented Hessian, structure from ref [1], eqn. (56)
        aug_H = np.zeros(shape=(h_n + 1, h_n + 1))

        aug_H[:h_n, :h_n] = self._coords.h
        aug_H[-1, :h_n] = self._coords.g
        aug_H[:h_n, -1] = self._coords.g

        aug_H_lmda, aug_H_v = np.linalg.eigh(aug_H)
        # A RF step uses the eigenvector corresponding to the lowest non zero
        # eigenvalue
        mode = np.where(np.abs(aug_H_lmda) > 1E-8)[0][0]
        logger.info(f'Stepping along mode: {mode}')

        # and the step scaled by the final element of the eigenvector
        delta_s = aug_H_v[:-1, mode] / aug_H_v[-1, mode]
        max_step_component = np.max(np.abs(delta_s))

        if max_step_component > self.alpha:
            logger.warning(f'Maximum component of the step '
                           f'{max_step_component:.4} Å > {self.alpha:.4f} '
                           f'Å. Scaling down')
            delta_s *= self.alpha / max_step_component

        self._coords = self._coords + delta_s

    def _initialise_run(self) -> None:
        """
        Initialise running an
        """
        self._coords = CartesianCoordinates(self._species.coordinates).to('dic')

        # TODO - initial Hessian estimate from low-level method
        self._coords.h = np.eye(len(self._coords))

        self._update_gradient_and_energy()

