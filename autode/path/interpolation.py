"""
Routines for interpolation of a path or series of images
"""
from typing import List, Optional, Union, Sequence, TYPE_CHECKING
import numpy as np
from math import sqrt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import minimize

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.values import Energy
    from scipy.interpolate import PPoly


class PathSpline:
    """
    Smooth cubic spline interpolation through a path i.e. a series
    of images, or coordinates. Optionally also fits the energy.
    """

    def __init__(
        self,
        coords_list: Sequence[np.ndarray],
        energies: Optional[Sequence[Union[float, "Energy"]]] = None,
    ):
        """
        Initialise a spline representation of path from list of coordinates
        and energies, if provided

        Args:
            coords_list (Sequence[np.ndarray]): List of coordinates
            energies (Sequence[float|Energy]): List of energies
        """
        # cast all coordinates into flat arrays, and check size
        coords_list = [np.array(coords).flatten() for coords in coords_list]
        assert all(
            coords.shape == coords_list[0].shape for coords in coords_list
        )

        self._path_spline = self._spline_from_coords(coords_list)
        self._energy_spline = None

        if energies is not None:
            self.fit_energies(energies)

    @staticmethod
    def _spline_from_coords(coords_list: Sequence[np.ndarray]):
        """
        Obtain a cubic spline from a set of flat coordinates

        Args:
            coords_list (Sequence[np.ndarray]): List of flat arrays, all
                                            must have same dimensions

        Returns:
            (PPoly): The fitted path spline
        """
        # Estimate normalised distances by adjacent Euclidean distances
        distances = [
            np.linalg.norm(coords_list[idx + 1] - coords_list[idx])
            for idx in range(len(coords_list) - 1)
        ]
        path_distances = [0] + list(np.cumsum(distances))
        max_dist = max(path_distances)
        path_distances = [dist / max_dist for dist in path_distances]
        coords_data = np.array(coords_list).transpose()
        return CubicSpline(x=path_distances, y=coords_data, axis=0)

    @property
    def path_distances(self) -> List[float]:
        assert self._path_spline is not None
        return list(self._path_spline.x)

    def fit_energies(self, energies: Sequence[Union[float, "Energy"]]) -> None:
        """
        Fit the energy spline based on the distances of
        the current path spline, and supplied energy values.
        Will overwrite any energies used during init.

        Args:
            energies (list[float|Energy]):
        """
        energies = [float(en) for en in energies]
        self._energy_spline = CubicSpline(
            x=self.path_distances,
            y=energies,
        )
        return None

    @classmethod
    def from_species_list(
        cls, species_list: Sequence["Species"], fit_energy=False
    ):
        """
        Obtain a cubic spline from a list of species.

        Args:
            species_list (Sequence[Species]): The list of species in the path,
                                in the order that they appear
            fit_energy (bool): Whether to use the energies from the species
                               to fit the energy spline

        Returns:
            (PathSpline):
        """
        coords_list = [
            np.array(mol.coordinates).flatten() for mol in species_list
        ]

        energies = None

        if fit_energy:
            energies = [mol.energy for mol in species_list]
            assert all(en is not None for en in energies)

        return cls(coords_list=coords_list, energies=energies)  # type: ignore

    def path_integral(self, l_bound: float, u_bound: float) -> float:
        """
        Integrate the parametric spline to obtain the length of the
        path, in a given range

        Args:
            l_bound (float): Lower bound of integration
            u_bound (float): Upper bound of integration

        Returns:
            (float): The path length
        """
        deriv = self._path_spline.derivative()
        assert deriv(l_bound).shape[0] > 1

        if l_bound > u_bound:
            l_bound, u_bound = u_bound, l_bound

        # TODO: check this formula again
        def dpath(t):
            return sqrt(np.sum(np.square(deriv(t))))

        path_length = quad(
            func=dpath,
            a=l_bound,
            b=u_bound,
            epsabs=1.0e-5,
        )

        return path_length[0]

    def integrate_upto_length(self, span, x0, sol_guess) -> float:
        """
        Solve the value of x for which, path integral from x0 to x will
        be equal to the given length.

        Args:
            span (float): The specified length
            x0 (float): Initial value
            sol_guess (float): Guess for the solution

        Returns:
            (float): The solution
        """
        res = minimize(
            fun=lambda x: self.path_integral(float(x), x0) - span,
            x0=np.array([sol_guess]),
            method="BFGS",
            tol=1.0e-5,
        )

        return float(res.x)

    def peak_x(self, l_bound: float, u_bound: float) -> Optional[float]:
        """
        Get the peak of the path within a given range,
        by using the energy spline

        Args:
            l_bound (float): Lower bound of range
            u_bound (float): Upper bound of range

        Returns:
            (float|None): Position of the peak, None if not found
        """
        if self._energy_spline is None:
            raise RuntimeError(
                "Energy spline must be fitted before calling peak_x()"
            )
        # cast into proper types
        l_bound = float(l_bound)
        u_bound = float(u_bound)

        deriv = self._energy_spline.derivative()
        # Obtain the roots of first derivative
        roots = deriv.roots(discontinuity=False, extrapolate=False)
        roots = roots[(roots < u_bound) & (roots > l_bound)]

        all_possible_points = [u_bound, l_bound] + list(roots)
        values = []
        for x in all_possible_points:
            # get the predicted energy from spline
            values.append(self._energy_spline(x)[-1])

        # Extreme value theorem means that inside a bound, there
        # must be a highest and lowest point on a continuous function
        # So, the highest point must be a maximum (within bounds)
        peak = np.argmax(values)
        if peak in [0, 1]:
            # means the highest point is on one of the bounds i.e. no peak
            return None
        else:
            return all_possible_points[peak]
