"""
Coordinates for dimer optimisations. Notation follows:
[1] https://aip.scitation.org/doi/pdf/10.1063/1.2815812
"""
import numpy as np
from typing import Union, Sequence, Optional
from autode.opt.coordinates.base import OptCoordinates
from autode.log import logger


class DimerCoordinates(OptCoordinates):

    def __new__(cls,
                input_array: Union[Sequence, np.ndarray],
                units:       Union[str, 'autode.units.Unit'] = 'Å'
                ) -> 'OptCoordinates':
        """New instance of these coordinates"""

        arr = super().__new__(cls, np.array(input_array), units)

        if arr.ndim != 2 or arr.shape[0] != 3:
            raise ValueError('Dimer coordinates must ')

        arr._e = None    # Energy
        arr._d = 0.0     # Translation distance
        arr._phi = 0.0   # Rotation amount

        """
        Compared to standard Cartesian coordinates these arrays have and
        additional dimension for the two end points of the dimer.
        """
        arr._g = None    # Gradient: {dE/dX_0, dE/dX_1, dE/dx_2}
        arr._h = None    # Hessian: {d2E/dXdY_0, d2E/dXdY_1, d2E/dXdY_2}

        return arr

    def __array_finalize__(self, obj: 'OptCoordinates') -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        for attr in ('units', '_e', '_g', '_h'):
            self.__dict__[attr] = getattr(obj, attr, None)

        return None

    @classmethod
    def from_species(cls,
                     species1: 'autode.species.Species',
                     species2: 'autode.species.Species'
                     ) -> 'DimerCoordinates':
        """
        Initialise a set of DimerCoordinates from two species, i.e. those
        either side of the saddle point.
        """
        if not species1.has_identical_composition_as(species2):
            raise ValueError('Cannot form a set of dimer coordinates from two '
                             'species with a different number of atoms')

        coords = cls(np.stack((np.zeros(3*species1.n_atoms),
                               np.array(species1.coordinates).flatten(),
                               np.array(species2.coordinates).flatten()),
                              axis=0))
        return coords

    def _update_g_from_cart_g(self,
                              arr: Optional['autode.values.Gradient']) -> None:
        raise NotImplementedError('Cannot update the gradient - indeterminate '
                                  'point in the dimer')

    def _update_h_from_cart_h(self,
                              arr: Optional['autode.values.Hessian']) -> None:
        logger.warning('Dimer does not require Hessians - skipping')
        return None

    def __repr__(self) -> str:
        return f'Dimer Coordinates({np.ndarray.__str__(self)} {self.units.name})'

    def to(self, *args, **kwargs) -> 'OptCoordinates':
        raise NotImplementedError('Cannot convert dimer coordinates to other '
                                  'types')

    def iadd(self, value: np.ndarray) -> 'OptCoordinates':
        return np.ndarray.__iadd__(self, value)

    @property
    def x0(self) -> np.ndarray:
        """Midpoint of the dimer"""
        return (self.x1 + self.x2) / 2.0

    @property
    def x1(self) -> np.ndarray:
        """Coordinates on the 'left' side of the dimer"""
        return np.array(self)[1, :]

    @property
    def x2(self) -> np.ndarray:
        """Coordinates on the 'right' side of the dimer"""
        return np.array(self)[2, :]

    def _g_vec(self, idx: int) -> np.ndarray:
        if self._g is None:
            raise RuntimeError('Cannot get the gradient')

        return self._g[idx, :]

    @property
    def g0(self) -> np.ndarray:
        """Gradient at the midpoint of the dimer"""
        return self._g_vec(0)

    @property
    def g1(self) -> np.ndarray:
        """Gradient on the 'left' side of the dimer"""
        return self._g_vec(1)

    @property
    def g2(self) -> np.ndarray:
        """Gradient on the 'right' side of the dimer"""
        return self._g_vec(1)

    @property
    def tau(self) -> np.ndarray:
        """Direction between the two ends of the dimer (τ)"""
        return (self.x1 - self.x2) / 2.0

    @property
    def tau_hat(self) -> np.ndarray:
        """Normalised direction between the two ends of the dimer"""
        tau = self.tau
        return tau / np.linalg.norm(tau)

    @property
    def f_r(self) -> np.ndarray:
        """Rotational force F_R. eqn. 3 in ref. [1]"""
        tau_hat = self.tau_hat
        x = 2.0 * (self.g1 - self.g0)

        return -x + np.dot(x, tau_hat) * tau_hat

    @property
    def f_t(self) -> np.ndarray:
        """Translational force F_T, eqn. 2 in ref. [1]"""
        g0, tau_hat = self.g0, self.tau_hat

        return -g0 + 2.0 * np.dot(g0, tau_hat) * tau_hat

    @property
    def delta(self) -> float:
        """Distance between the dimer point, Δ"""
        delta = np.linalg.norm(self.x1 - self.x2) / 2.0

        if np.isclose(delta, 0.0):
            raise RuntimeError('Zero distance between the dimer points')

        return delta

    @property
    def theta(self) -> np.ndarray:
        """Rotation direction Θ, calculated using steepest descent"""
        f_r = self.f_r

        # F_R / |F_R| with a small jitter to prevent division by zero
        return f_r / (np.linalg.norm(f_r) + 1E-8)

    @property
    def c(self) -> float:
        """Curvature of the PES, C_τ.  eqn. 4 in ref [1]"""
        return np.dot((self.g1 - self.g0), self.tau_hat) / self.delta

    @property
    def dc_dphi(self) -> float:
        """dC_τ/dϕ eqn. 6 in ref [1] """
        return 2.0 * np.dot((self.g1 - self.g0), self.theta) / self.delta
