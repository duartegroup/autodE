import numpy as np
from abc import ABC
from typing import Type

from autode.log import logger
from autode.values import GradientRMS, PotentialEnergy
from autode.opt.optimisers.base import NDOptimiser
from autode.opt.optimisers.hessian_update import BFGSUpdate, NullUpdate
from autode.opt.optimisers.line_search import (
    LineSearchOptimiser,
    ArmijoLineSearch,
)


class BFGSOptimiser(NDOptimiser, ABC):
    def __init__(
        self,
        maxiter: int,
        gtol: GradientRMS,
        etol: PotentialEnergy,
        init_alpha: float = 1.0,
        line_search_type: Type[LineSearchOptimiser] = ArmijoLineSearch,
        **kwargs,
    ):
        """
        Broyden–Fletcher–Goldfarb–Shanno optimiser. Implementation taken
        from: https://tinyurl.com/526yymsw

        ----------------------------------------------------------------------
        Arguments:
            init_alpha (float): Length of the initial step to take in the line
                               search. Units of distance

        See Also:

            :py:meth:`NDOptimiser <autode.opt.optimisers.base.NDOptimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, gtol=gtol, etol=etol, **kwargs)

        self._line_search_type = line_search_type
        self._hessian_update_types = [BFGSUpdate, NullUpdate]
        self._alpha = init_alpha

    def _step(self) -> None:
        r"""
        Perform a BFGS step. Requires an initial guess of the Hessian matrix
        i.e. (self._coords.h must be defined). Steps follow:

        1. Determine the inverse Hessian: :py:meth:`h_inv <autode.opt.
        optimisers.hessian_update.HessianUpdater>`

        2. Determine the search direction with:

        .. math::

             \boldsymbol{p}_k = - H_k^{-1} \nabla E

        where H is the Hessian matrix, p is the search direction and
        :math:`\nabla E` is the gradient of the energy with respect to the
        coordinates. On the first iteration :math:`H_0` is either the true
        or exact Hessian.

        3. Performing a (in)exact line search to obtain a suitable step size

        .. math::

            \alpha_k = \text{arg min} E(X_k + \alpha \boldsymbol{p}_k)

        and setting :math:`s_k = \alpha \boldsymbol{p}_k`, and updating the
        positions accordingly (:math:`X_{k+1} = X_{k} + s_k`)
        """
        assert (
            self._coords is not None
            and self._coords.g is not None
            and self._species is not None
            and self._method is not None
        )

        self._coords.h_inv = self._updated_h_inv()
        p = np.matmul(self._coords.h_inv, -self._coords.g)

        logger.info("Performing a line search")
        ls = self._line_search_type(
            direction=p, init_alpha=self._alpha, coords=self._coords.copy()
        )

        ls.run(self._species, self._method, n_cores=self._n_cores)
        assert ls.minimum_e_coords is not None

        self._coords = ls.minimum_e_coords.copy()
        return None
