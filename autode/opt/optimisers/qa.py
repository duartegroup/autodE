"""
Constrained optimisation with quadratic trust radius model

Also known as Quadratic Approximation (QA) or Trust-Radius Model (TRM)

References:
[1] P. Culot et al. Theor. Chim. Acta, 82, 1992, 189-205
[2] T. Helgaker, Chem. Phys. Lett., 182(5), 1991, 503-510
[3] J. T. Golab et al. Chem. Phys., 78, 1983, 175-199
[4] R. Fletcher, Practical Methods of Optimization, Wiley, Chichester, 1981
"""
import numpy as np
from scipy.optimize import root_scalar

from autode.log import logger
from autode.opt.optimisers.crfo import CRFOptimiser
from autode.exceptions import OptimiserStepError


class QAOptimiser(CRFOptimiser):
    """Quadratic trust-radius optimiser in delocalised internal coordinates"""

    def _step(self) -> None:
        """Trust radius step"""
        assert self._coords is not None, "Must have coords to take a step"

        if self.iteration != 0:
            self._coords.update_h_from_old_h(
                self._history.penultimate, self._hessian_update_types
            )
        assert self._coords.h is not None

        self._update_trust_radius()
        self._log_constrained_opt_progress()

        n = len(self._coords)

        # Take RFO step if within trust radius
        delta_s_rfo = self._get_rfo_step()
        if np.linalg.norm(delta_s_rfo[:n]) <= self.alpha:
            logger.info("Taking an RFO step")
            self._take_step_within_max_move(delta_s_rfo)
            return None

        # otherwise use QA step within trust
        try:
            delta_s_qa = self._get_qa_step()
            logger.info("Taking a QA step within trust radius")
            self._take_step_within_max_move(delta_s_qa)
            return None

        # if QA fails, used scaled RFO step
        except OptimiserStepError as exc:
            logger.info(f"QA step failed: {str(exc)}, using scaled RFO step")
            factor = self.alpha / np.linalg.norm(delta_s_rfo[:n])
            self._take_step_within_max_move(delta_s_rfo * factor)
            return None

    def _get_qa_step(self):
        """
        Calculate the QA step within trust radius for the current
        set of coordinates

        Returns:
            (np.ndarray): The trust radius step
        """
        n, m = len(self._coords), self._coords.n_constraints
        idxs = self._coords.active_indexes

        def shifted_newton_step(hess, grad, lmda, check=False):
            """
            Level-shifted Newton step (H-λI)^-1 . g
            optional check of Hessian eigenvalue structure
            """
            hess = hess - lmda * np.eye(hess.shape[0])
            # no shift on constraints
            for i in range(m):
                hess[-m + i, -m + i] = 0.0
            full_step = np.zeros_like(grad)
            hess = hess[:, idxs][idxs, :]
            grad = grad[idxs]
            if check:
                self._check_shifted_hessian_has_correct_struct(hess)
            qa_step = -np.matmul(np.linalg.inv(hess), grad)
            full_step[idxs] = qa_step
            return full_step

        def qa_step_error(lmda):
            """Error in step size"""
            ds = shifted_newton_step(self._coords.h, self._coords.g, lmda)
            ds_atoms = ds[:n]
            return np.linalg.norm(ds_atoms) - self.alpha

        # if molar Hessian +ve definite & step within trust use simple qN
        min_b = self._coords.min_eigval
        if min_b > 0 and qa_step_error(0.0) <= 0.0:
            return shifted_newton_step(
                self._coords.h, self._coords.g, 0.0, True
            )

        # Find λ in range (-inf, b)
        for k in range(500):
            right_bound = min_b - 0.5**k
            if qa_step_error(right_bound) > 0:
                break
        assert qa_step_error(right_bound) > 0

        for k in range(-6, 10):
            left_bound = right_bound - 2**k
            if qa_step_error(left_bound) < 0:
                break
        if not qa_step_error(left_bound) < 0:
            raise OptimiserStepError("Unable to find bounds for root search")

        res = root_scalar(f=qa_step_error, bracket=[left_bound, right_bound])
        if (not res.converged) or (res.root >= min_b):
            raise OptimiserStepError("QA root search failed")

        logger.info(f"Calculated QA λ = {res.root:.4f}")
        return shifted_newton_step(
            self._coords.h, self._coords.g, res.root, True
        )
