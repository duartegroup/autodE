r"""
Trust region methods for performing optimisations, in contrast to line
search methods the direction within the 1D problem is optimised rather than
the distance in a particular direction. The sub-problem is:

.. math::

    \min(m_k(p)) = E_k + g_k^T p + \frac{1}{2}p^T H_k p
    \quad : \quad ||p|| \le \alpha_k

where :math:`p` is the search direction, :math:`E_k` is the energy at an
iteration :math:`k`, :math:`g_k` is the gradient and :math:`H` is the Hessian
and :math:`alpha_k` is the trust radius at step k. Notation follows
https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
with :math:`\Delta \equiv \alpha`
"""
import numpy as np
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from autode.log import logger
from autode.values import GradientRMS, PotentialEnergy
from autode.opt.optimisers.base import NDOptimiser
from autode.opt import CartesianCoordinates

if TYPE_CHECKING:
    from autode.opt.coordinates import OptCoordinates
    from autode.species.species import Species
    from autode.wrappers.methods import Method


class TrustRegionOptimiser(NDOptimiser, ABC):
    def __init__(
        self,
        maxiter: int,
        gtol: GradientRMS,
        etol: PotentialEnergy,
        trust_radius: float,
        coords: Optional["OptCoordinates"] = None,
        max_trust_radius: Optional[float] = None,
        eta_1: float = 0.1,
        eta_2: float = 0.25,
        eta_3: float = 0.75,
        t_1: float = 0.25,
        t_2: float = 2.0,
        **kwargs,
    ):
        """Trust radius optimiser"""
        super().__init__(
            maxiter=maxiter, etol=etol, gtol=gtol, coords=coords, **kwargs
        )

        self.alpha = trust_radius
        self.alpha_max = (
            max_trust_radius
            if max_trust_radius is not None
            else 10 * trust_radius
        )

        # Parameters for the TR optimiser
        self._eta = _Eta(eta_1, eta_2, eta_3)
        self._t = _T(t_1, t_2)

        self.m: Optional[float] = None  # Energy estimate
        self.p: Optional[np.ndarray] = None  # Direction

    @classmethod
    def optimise(
        cls,
        species: "Species",
        method: "Method",
        n_cores: Optional[int] = None,
        coords: Optional["OptCoordinates"] = None,
        maxiter: int = 5,
        gtol: GradientRMS = GradientRMS(1e-3, "Ha Å-1"),
        etol: PotentialEnergy = PotentialEnergy(1e-4, "Ha"),
        trust_radius: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Construct and optimiser using a trust region optimiser
        """

        optimiser = cls(
            maxiter=maxiter,
            gtol=gtol,
            etol=etol,
            trust_radius=trust_radius,
            coords=coords,
        )

        optimiser.run(species, method, n_cores=n_cores)

        return None

    def _step(self) -> None:
        """
        Perform a TR step based on a solution to the 'sub-problem' to find a
        direction which to step in, the distance for which is fixed by the
        current trust region (alpha)
        """
        assert self._coords is not None

        rho = self.rho
        self._solve_subproblem()

        e, g, h, p = self._coords.e, self._coords.g, self._coords.h, self.p
        self.m = e + np.dot(g, p) + 0.5 * np.dot(p, np.matmul(h, p))

        if self.iteration == 0:
            # First iteration, so take a normal step
            self._coords = self._coords + self.p
            return

        if rho < self._eta[2]:
            self.alpha *= self._t[1]

        else:
            if rho > self._eta[3] and self._step_was_close_to_max:
                self.alpha = min(self._t[2] * self.alpha, self.alpha_max)

            else:
                pass  # No updated required: α_k+1 = α_k

        if rho > self._eta[1]:
            self._coords = self._coords + self.p

        else:
            logger.warning(
                "Trust radius step did not result in a satisfactory"
                " reduction. Not taking a step"
            )
            self._coords = self._coords.copy()

        return None

    @property
    def _step_was_close_to_max(self) -> bool:
        """
        Is the current step close to the maximum allowed?

        -----------------------------------------------------------------------
        Returns:
            (bool): |p| ~ α_max
        """
        return np.allclose(np.linalg.norm(self.p), self.alpha)

    @abstractmethod
    def _solve_subproblem(self) -> None:
        """Solve the TR 'subproblem' for the ideal step to use"""

    @abstractmethod
    def _update_hessian(self) -> None:
        """Solve the TR 'subproblem' for the ideal step to use"""

    def _update_gradient_and_energy(self) -> None:
        """
        Update the gradient and energy, along with a couple of other
        properties, derived from the energy and gradient
        """
        super()._update_gradient_and_energy()
        self._update_hessian()
        return None

    @property
    def rho(self) -> float:
        """
        Calculate ρ, the ratio of the actual and predicted reductions for the
        previous step

        -----------------------------------------------------------------------
        Returns:
            (float): ρ
        """

        if self.iteration == 0:
            logger.warning(
                "ρ is unknown for the 0th iteration with only one "
                "energy and gradient having been evaluated"
            )
            return np.inf

        if not self._last_step_updated_coordinates:
            logger.warning(
                f"Step {self.iteration} did not update the "
                "coordinates, using ρ = 0.5"
            )
            return 0.5

        if self.m is None:
            raise RuntimeError("Predicted energy update (m) undefined")

        penultimate_energy = self._history.penultimate.e
        final_energy = self._history.final.e
        assert final_energy is not None and penultimate_energy is not None

        true_diff: PotentialEnergy = penultimate_energy - final_energy
        predicted_diff: PotentialEnergy = penultimate_energy - self.m

        return true_diff / predicted_diff

    @property
    def _last_step_updated_coordinates(self) -> bool:
        """
        Did the last step in the optimiser update the coordinates

        Returns:
            (bool): If the last step updated the coordinates
        """
        dx = np.array(self._history.final, copy=True) - np.array(
            self._history.penultimate, copy=True
        )

        return np.linalg.norm(dx) > 1e-10


class CauchyTROptimiser(TrustRegionOptimiser):
    """Most simple trust-radius optimiser, solving the subproblem with a
    cauchy point calculation"""

    def _initialise_run(self) -> None:
        """Initialise a TR optimiser, so it can take the first step"""

        if self._coords is None:
            assert self._species is not None
            self._coords = CartesianCoordinates(self._species.coordinates)

        self._update_gradient_and_energy()
        self._solve_subproblem()
        return None

    def _update_hessian(self) -> None:
        """Hessian is always the identity matrix"""
        assert self._coords is not None
        self._coords.h = np.eye(len(self._coords))
        return None

    def _solve_subproblem(self) -> None:
        r"""
        Solve for the optimum direction by a Cauchy point calculation

        .. math::

            \tau =
            \begin{cases}
            1 \qquad \text{ if } g^T H g <= 0\\
            \min\left( \frac{|g|^3}{\alpha g^T H g}, 1\right)
            \qquad \text{otherwise}
            \end{cases}

        and

        .. math::

            p = -\tau \frac{\alpha}{|g|} g

        """
        assert self._coords is not None
        g = self._coords.g

        self.p = -self.tau * (self.alpha / np.linalg.norm(g)) * g

        return None

    @property
    def tau(self) -> float:
        """
        Calculate τ

        ----------------------------------------------------------------------
        Returns:
            (float): τ
        """
        assert self._coords is not None

        e, g, h = self._coords.e, self._coords.g, self._coords.h
        g_h_g = np.dot(g, np.matmul(h, g))

        if g_h_g <= 0:
            return 1.0
        else:
            return min((np.linalg.norm(g) ** 3 / (self.alpha * g_h_g), 1.0))


class DoglegTROptimiser(CauchyTROptimiser):
    """Dogleg Method for solving the subproblem"""

    def _solve_subproblem(self) -> None:
        """
        Solve the subproblem to generate a dogleg step
        """
        assert self._coords is not None
        tau, g, h = self.tau, self._coords.g, self._coords.h

        p_u = -(np.dot(g, g) / np.dot(g, np.matmul(h, g))) * g

        if 0 < tau <= 1:
            self.p = tau * p_u

        elif 1 < tau <= 2:
            raise NotImplementedError
            # self.p = tau * p_u + (tau - 1) * (p_b - p_u)

        else:
            raise RuntimeError(f"τ = {tau} was outside the acceptable region")

        return None


class CGSteihaugTROptimiser(TrustRegionOptimiser):
    """Conjugate Gradient Steihaung's Method"""

    coordinate_type = CartesianCoordinates

    def __init__(self, *args, epsilon: float = 0.001, **kwargs):
        """
        CG trust region optimiser

        -----------------------------------------------------------------------
        Arguments:
            *args: Arguments to pass to TrustRegionOptimiser

            epsilon: ε parameter

            **kwargs: Keyword arguments to pass to TrustRegionOptimiser
        """
        super().__init__(*args, **kwargs)

        self.epsilon = epsilon

    def _initialise_run(self) -> None:
        """Initialise a TR optimiser, so it can take the first step"""

        if self._coords is None:
            assert self._species is not None
            self._coords = self.coordinate_type(self._species.coordinates)

        self._update_gradient_and_energy()
        self._solve_subproblem()
        return None

    def _update_hessian(self) -> None:
        """Hessian is always the identity matrix"""
        assert self._coords is not None
        self._coords.h = np.eye(len(self._coords))
        return None

    def _discrete_optimised_p(self, z, d) -> np.ndarray:
        """
        Optimise the direction of the step to take by performing an exact line
        search over τ as to reduce the value of 'm' as much as possible, where
        m is the predicted reduction in the energy by performing this step

        -----------------------------------------------------------------------
        Arguments:
            z: Current estimate of the step
            d: Direction to move the step in

        Returns:
            (np.ndarray): Step direction (p)
        """
        assert self._coords is not None

        e, g, h = self._coords.e, self._coords.g, self._coords.h
        tau_arr, m_arr = np.linspace(0, 10, num=1000), []

        for tau in tau_arr:
            p = z + tau * d
            m = e + np.dot(g, p) + 0.5 * np.dot(p, np.matmul(h, p))

            m_arr.append(m)

        min_m_tau = tau_arr[np.argmin(m_arr)]
        logger.info(f"Optimised τ={min_m_tau:.6f}")

        p = z + min_m_tau * d
        step_length = np.linalg.norm(p)

        if step_length > self.alpha:
            logger.warning(
                f"Step size {step_length} was too large, " f"scaling down"
            )
            p *= self.alpha / step_length

        return p

    def _solve_subproblem(self) -> None:
        """
        Solve the subproblem for a direction
        """
        assert self._coords is not None

        h = self._coords.h
        z, r = 0.0, np.array(self._coords.g, copy=True)
        d = -r.copy()

        if np.linalg.norm(r) < self.epsilon:
            self.p = np.zeros_like(r)
            return

        for j in range(100):
            if np.dot(d, np.matmul(h, d)) <= 0:
                self.p = self._discrete_optimised_p(z, d)
                return

            alpha = np.dot(r, r) / (np.dot(d, np.matmul(h, d)))
            z += alpha * d

            if np.linalg.norm(z) >= self.alpha:
                self.p = self._discrete_optimised_p(z, d)
                return

            r_old = r.copy()
            r += alpha * np.matmul(h, d)

            if np.linalg.norm(r) < self.epsilon:
                self.p = z
                return

            beta = np.dot(r, r) / np.dot(r_old, r_old)
            d = -r + beta * d

        raise RuntimeError("Failed to converge CG trust region solve")


class _ParametersIndexedFromOne:
    def __getitem__(self, item):
        """Internal array is indexed from 1"""
        return self._arr[item - 1]

    def __init__(self, *args):
        """Scalar float arguments"""
        self._arr = [float(arg) for arg in args]


class _Eta(_ParametersIndexedFromOne):
    """η parameters in the TR optimisers"""

    def __init__(self, p1, p2, p3):
        super().__init__(p1, p2, p3)


class _T(_ParametersIndexedFromOne):
    """t parameters in the TR optimisers"""

    def __init__(self, p1, p2):
        super().__init__(p1, p2)
