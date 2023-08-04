"""
Routines for interpolation of a path or series of images
"""
from typing import List, Optional, Tuple, Sequence, TYPE_CHECKING
import numpy as np
from math import sqrt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import minimize

if TYPE_CHECKING:
    from autode.species.species import Species
    from scipy.interpolate import PPoly


class PathSpline(CubicSpline):
    """
    Smooth cubic spline interpolation through a path i.e. a series
    of images, or coordinates. Optionally also fits the energy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.energy_spline = None

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
        # Estimate normalised distances by adjacent Euclidean distances
        distances = [
            np.linalg.norm(coords_list[idx + 1] - coords_list[idx])
            for idx in range(len(coords_list) - 1)
        ]
        path_distances = [0] + list(np.cumsum(distances))
        max_dist = max(path_distances)
        path_distances = [dist / max_dist for dist in path_distances]

        coords_data = np.array(coords_list).transpose()
        return cls(x=path_distances, y=coords_data, axis=0)

    def path_integral(self, l_bound: float, u_bound: float) -> float:
        deriv = self.derivative()
        assert deriv(l_bound).shape[0] > 1
        assert l_bound < u_bound

        # TODO: check this formula again
        def dpath(t):
            return sqrt(np.sum(np.square(deriv(t))))

        path_length = quad(
            func=dpath,
            a=l_bound,
            b=u_bound,
            epsabs=1.0e-5,
        )

        return float(path_length)

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
        if x0 < sol_guess:
            func = lambda x: self.path_integral(x0, float(x)) - span
        else:
            func = lambda x: self.path_integral(float(x), x0) - span

        res = minimize(
            fun=func,
            x0=np.array([sol_guess]),
            method="BFGS",
            tol=1.0e-5,
        )

        return float(res.x)


def _get_1d_spline_peak(
    spline: "PPoly", l_bound: float, u_bound: float
) -> Optional[float]:
    """
    Get the peak point of a scipy 1d spline, within a
    given range

    Args:
        spline (PPoly): The spline, must be 1D, i.e. one output
        l_bound (float): Lower bound of range
        u_bound (float): Upper bound of range

    Returns:
        (float|None): Position of the peak, None if not found
    """
    # cast into proper types
    l_bound = float(l_bound)
    u_bound = float(u_bound)

    deriv = spline.derivative()
    # Obtain the roots of first derivative
    roots = deriv.roots(discontinuity=False, extrapolate=False)
    roots = roots[(roots < u_bound) & (roots > l_bound)]

    all_possible_points = [u_bound, l_bound] + list(roots)
    values = []
    for x in all_possible_points:
        # get the predicted energy from spline
        values.append(spline(x)[-1])

    # Extreme value theorem means that inside a bound, there
    # must be a highest and lowest point on a continuous function
    # So, the highest point must be a maximum (within bounds)
    peak = np.argmax(values)
    if peak in [0, 1]:
        # means the highest point is on one of the bounds i.e. no peak
        return None
    else:
        return all_possible_points[peak]
