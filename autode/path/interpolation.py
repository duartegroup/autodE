"""
Routines for interpolation of a path or series of images
"""
from typing import List, Optional, Union, Sequence, TYPE_CHECKING
import numpy as np
from math import sqrt
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.integrate import quad
from scipy.optimize import root_scalar

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.values import Energy
    from scipy.interpolate import PPoly


class CubicPathSpline:
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
        self._energy_spline: Optional["PPoly"] = None

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
        distances = [0.0] + [
            np.linalg.norm(coords_list[idx + 1] - coords_list[idx])
            for idx in range(len(coords_list) - 1)
        ]
        path_distances = np.cumsum(distances)
        path_distances /= max(path_distances)
        coords_data = np.array(coords_list)
        return CubicSpline(x=path_distances, y=coords_data, axis=0)

    @property
    def path_distances(self) -> List[float]:
        """
        Locations of each point in the current spline, according
        to normalised Euclidean distances (chord-length parameterisation)

        Returns:
            (list[float]):
        """
        assert self._path_spline is not None
        return list(self._path_spline.x)

    def fit_energies(self, energies: Sequence[Union[float, "Energy"]]) -> None:
        """
        Fit the energy spline based on the distances of
        the current path spline, and supplied energy values.
        Will overwrite any energies used during init.

        Args:
            energies (Sequence[float|Energy]):
        """
        energies = [float(energy) for energy in energies]
        self._energy_spline = CubicSpline(
            x=self.path_distances,
            y=energies,
        )
        return None

    def coords_at(self, path_distance: float) -> np.ndarray:
        """Spline-predicted coordinates at a point"""
        return self._path_spline(path_distance)

    def energy_at(self, path_distance: float) -> float:
        """Spline-predicted energy at a point"""
        if self._energy_spline is None:
            raise RuntimeError(
                "Must have fitted energies before calling energy_at()"
            )
        return self._energy_spline(path_distance)

    @classmethod
    def from_species_list(
        cls, species_list: Sequence["Species"]
    ) -> "CubicPathSpline":
        """
        Obtain a cubic spline from a list of species. Will fit energies if they
        are available on all species provided.

        Args:
            species_list (Sequence[Species]): The list of species in the path,
                                in the order that they appear

        Returns:
            (PathSpline):
        """
        coords_list = [
            np.array(mol.coordinates).flatten() for mol in species_list
        ]

        energies: Optional[list] = [mol.energy for mol in species_list]
        if any(mol.energy is None for mol in species_list):
            energies = None

        return cls(coords_list=coords_list, energies=energies)  # type: ignore

    def path_integral(
        self, l_bound: float = 0.0, u_bound: float = 1.0
    ) -> float:
        """
        Integrate the parametric spline to obtain the length of the
        path, in a given range. The bounds should be ideally in the
        range [0, 1], beyond that range the spline extrapolation is
        unreliable.

        Args:
            l_bound (float): Lower bound of integration
            u_bound (float): Upper bound of integration

        Returns:
            (float): The path length in the units of the coordinates fitted
        """
        deriv = self._path_spline.derivative()
        assert deriv(l_bound).shape[0] > 1

        assert l_bound < u_bound, "Lower bound must be less than upper bound"

        def dpath(t):
            return sqrt(np.sum(np.square(deriv(t))))

        path_length = quad(
            func=dpath,
            a=l_bound,
            b=u_bound,
            epsabs=1.0e-6,
            limit=100,
        )

        return path_length[0]

    def integrate_upto_length(self, span: float) -> float:
        """
        Solve the value of x for which path integral from 0 to x will
        be equal to the given length.

        Args:
            span (float): The specified length in the units of the
                          fitted coordinates (must be positive)

        Returns:
            (float): The solution
        """

        # Find bounds for root search
        def span_error(x):
            return self.path_integral(0, x) - span

        assert span > 0
        bracket_left = None
        bracket_right = None

        x_tmp = span
        for _ in range(500):
            if span_error(x_tmp) < 0:
                bracket_left = x_tmp
                x_tmp = x_tmp * 1.5
            else:
                bracket_right = x_tmp
                x_tmp = x_tmp * 0.5
            if bracket_left is not None and bracket_right is not None:
                break

        assert (
            bracket_left is not None and bracket_right is not None
        ), "Unable to find range for root search to integrate upto length"

        res = root_scalar(
            f=span_error,
            bracket=[bracket_left, bracket_right],
            method="brentq",
            xtol=1.0e-5,
        )

        assert res.converged, "Failed to integrate upto length!"
        return float(res.root)

    def energy_peak(
        self, l_bound: float = 0.0, u_bound: float = 1.0
    ) -> Optional[float]:
        """
        Get the peak of the path within a given range,
        by using the energy spline

        Args:
            l_bound (float): Lower bound of range
            u_bound (float): Upper bound of range

        Returns:
            (float|None): Position of the peak, None if not found

        Raises:
            RuntimeError: If energy was not fitted
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
            values.append(float(self._energy_spline(x)))

        # Extreme value theorem means that inside a bound, there
        # must be a highest and lowest point on a continuous function
        # So, the highest point must be a maximum (within bounds)
        peak = np.argmax(values)
        if peak in [0, 1]:
            # means the highest point is on one of the bounds i.e. no peak
            return None
        else:
            return all_possible_points[peak]
