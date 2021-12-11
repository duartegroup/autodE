import numpy as np
from abc import ABC
from typing import Type
from autode.log import logger
from autode.opt.optimisers.base import NDOptimiser
from autode.opt.optimisers.hessian_update import BFGSUpdate, NullUpdate
from autode.opt.optimisers.line_search import (LineSearchOptimiser,
                                               ArmijoLineSearch)


class BFGSOptimiser(NDOptimiser, ABC):

    def __init__(self,
                 maxiter:          int,
                 gtol:             'autode.values.GradientNorm',
                 etol:             'autode.values.PotentialEnergy',
                 init_alpha:       float = 1.0,
                 line_search_type: Type[LineSearchOptimiser] = ArmijoLineSearch,
                 **kwargs):
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
        self._h_update_types = [BFGSUpdate, NullUpdate]
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
        self._coords.h_inv = self._updated_h_inv()
        p = np.matmul(self._coords.h_inv, -self._coords.g)

        logger.info('Performing a line search')
        ls = self._line_search_type(direction=p,
                                    init_alpha=self._alpha,
                                    coords=self._coords.copy())

        ls.run(self._species, self._method, n_cores=self._n_cores)

        self._coords = ls.minimum_e_coords.copy()
        return None

    def _updated_h_inv(self) -> np.ndarray:
        r"""
        Update the inverse of the Hessian matrix :math:`H^{-1}` for the
        current set of coordinates. If the first iteration then use the true
        inverse of the (estimated) Hessian, otherwise update the inverse
        using a viable update strategy


        .. math::

            H_{l - 1} \rightarrow H_{l}

        """

        if self.iteration == 0:
            logger.info('First iteration so using exact inverse, H^-1')
            return np.linalg.inv(self._coords.h)

        coords_l, coords_k = self._coords, self._history.penultimate

        for update_type in self._h_update_types:
            updater = update_type(h_inv=coords_k.h_inv,
                                  s=coords_l - coords_k,
                                  y=coords_l.g - coords_k.g)

            if not updater.conditions_met:
                continue

            return updater.updated_h_inv

        raise RuntimeError('Could not update the inverse Hessian - no '
                           'suitable update strategies')
