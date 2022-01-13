"""Partitioned rational function optimisation"""
import numpy as np
from autode.log import logger
from autode.opt.coordinates import CartesianCoordinates
from autode.opt.optimisers.rfo import RFOOptimiser
from autode.opt.optimisers.hessian_update import BofillUpdate


class PRFOOptimiser(RFOOptimiser):

    def __init__(self,
                 init_alpha:    float = 0.1,
                 imag_mode_idx: int = 0,
                 *args,
                 **kwargs):
        """
        Partitioned rational function optimiser (PRFO) using a maximum step
        size of alpha trying to maximise along a mode while minimising along
        all others to locate a transition state (TS)

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Initial step size

            imag_mode_idx: Index of the imaginary mode to follow. Default = 0,
                           the most imaginary mode (i.e. most negative
                           eigenvalue)

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = float(init_alpha)
        self.imag_mode_idx = int(imag_mode_idx)

        self._hessian_update_types = [BofillUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""

        self._coords.h = self._updated_h()

        # Eigenvalues (\tilde{H}_kk in ref [1]) and eigenvectors (V in ref [1])
        # of the Hessian matrix
        lmda, v = np.linalg.eigh(self._coords.h)

        if np.min(lmda) > 0:
            raise RuntimeError('Hessian had no negative eigenvalues, cannot '
                               'follow to a TS')

        logger.info(f'Maximising along mode {self.imag_mode_idx} with '
                    f'λ={lmda[self.imag_mode_idx]:.4f}')

        # Gradient in the eigenbasis of the Hessian. egn 49 in ref. 50
        g_tilde = np.matmul(v.T, self._coords.g)

        # Initialised step in the Hessian eigenbasis
        s_tilde = np.zeros_like(g_tilde)

        # For a step in Cartesian coordinates the Hessian will have zero
        # eigenvalues for translation/rotation - keep track of them
        non_zero_lmda = np.where(np.abs(lmda) > 1E-8)[0]

        # Augmented Hessian 1 along the imaginary mode to maximise, with the
        # form (see eqn. 59 in ref [1]):
        #  (\tilde{H}_11  \tilde{g}_1) (\tilde{s}_1)  =      (\tilde{s}_1)
        #  (                         ) (           )  =  ν_R (           )
        #  (\tilde{g}_1        0     ) (    1      )         (     1     )
        #
        aug1 = np.array([[lmda[self.imag_mode_idx], g_tilde[self.imag_mode_idx]],
                         [g_tilde[self.imag_mode_idx], 0.0]])
        _, aug1_v = np.linalg.eigh(aug1)

        # component of the step along the imaginary mode is the first element
        # of the eigenvector with the largest eigenvalue (1), scaled by the
        # final element
        s_tilde[self.imag_mode_idx] = aug1_v[0, 1] / aug1_v[1, 1]

        # Augmented Hessian along all other modes with non-zero eigenvalues,
        # that are also not the imaginary mode to be followed
        non_mode_lmda = np.delete(non_zero_lmda, [self.imag_mode_idx])

        # see eqn. 60 in ref. [1] for the structure of this matrix!
        augn = np.diag(np.concatenate((lmda[non_mode_lmda], np.zeros(1))))
        augn[:-1, -1] = g_tilde[non_mode_lmda]
        augn[-1, :-1] = g_tilde[non_mode_lmda]

        _, augn_v = np.linalg.eigh(augn)

        # The step along all other components is then the all but the final
        # component of the eigenvector with the smallest eigenvalue (0)
        s_tilde[non_mode_lmda] = augn_v[:-1, 0] / augn_v[-1, 0]

        # Transform back from the eigenbasis with eqn. 52 in ref [1]
        delta_s = np.matmul(v, s_tilde)

        self._coords = self._coords + self._sanitised_step(delta_s)
        return None

    def _initialise_run(self) -> None:
        """
        Initialise running a partitioned rational function optimisation by
        setting the coordinates and Hessian
        """
        self._coords = CartesianCoordinates(self._species.coordinates).to('dic')
        self._update_hessian_gradient_and_energy()
        return None
