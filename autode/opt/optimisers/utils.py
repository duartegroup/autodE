"""
Various operations for coordinates and optimisers
"""
import numpy as np
from numpy.polynomial import Polynomial
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from autode.opt.coordinates.base import OptCoordinates


class TruncatedTaylor:
    """The truncated taylor surface from current grad and hessian"""

    def __init__(
        self,
        centre: Union["OptCoordinates", np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
    ):
        """
        Second-order Taylor expansion around a point

        Args:
            centre (OptCoordinates|np.ndarray): The coordinate point
            grad (np.ndarray): Gradient at that point
            hess (np.ndarray): Hessian at that point
        """
        self.centre = centre
        if hasattr(centre, "e") and centre.e is not None:
            self.e = centre.e
        else:
            # the energy can be relative and need not be absolute
            self.e = 0.0
        self.grad = grad
        self.hess = hess
        n_atoms = grad.shape[0]
        assert hess.shape == (n_atoms, n_atoms)

    def value(self, coords: np.ndarray) -> float:
        """Energy (or relative energy if point did not have energy)"""
        # E = E(0) + g^T . dx + 0.5 * dx^T. H. dx
        dx = (coords - self.centre).flatten()
        new_e = self.e + np.dot(self.grad, dx)
        new_e += 0.5 * np.linalg.multi_dot((dx, self.hess, dx))
        return new_e

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """Gradient at supplied coordinate"""
        # g = g(0) + H . dx
        dx = (coords - self.centre).flatten()
        new_g = self.grad + np.matmul(self.hess, dx)
        return new_g


def two_point_cubic_fit(
    e0: float, g0: float, e1: float, g1: float
) -> Polynomial:
    """
    Fit a general cubic equation with two points, using the
    energy and directional gradient at both points (assuming
    normalised distance between the two points).
    Equation: f(x) = d + cx + bx**2 + ax**3

    Args:
        e0:
        g0:
        e1:
        g1:

    Returns:
        (Polynomial): The fitted cubic polynomial
    """
    # f(0) = d; f(1) = a + b + c + d
    d = e0
    # f'(0) = c => a + b = f(1) - c - d
    c = g0
    a_b = e1 - c - d
    # f'(1) = 3a + 2b + c => 3a + 2b = f'(1) - c
    a3_2b = g1 - c
    a = a3_2b - 2 * a_b
    b = a_b - a
    return Polynomial([d, c, b, a])


def get_poly_extremum(
    poly: Polynomial, l_bound: float = 0.0, u_bound: float = 1.0, get_max=False
) -> Union[float, None]:
    """
    Obtain the maximum/minimum of a polynomial f(x), within
    two bounds. If there are multiple, return the highest/lowest
    respectively.

    Args:
        poly (Polynomial):
        l_bound (float):
        u_bound (float):
        get_max (bool): Maximum or minimum requested

    Returns:
        (float|None): The x value at min or max f(x), None
                    if not found or not within bounds
    """
    # points with derivative 0 are critical points
    crit_points = poly.deriv().roots()

    if l_bound > u_bound:
        u_bound, l_bound = l_bound, u_bound
    crit_points = crit_points[crit_points < u_bound]
    crit_points = crit_points[crit_points > l_bound]

    if len(crit_points) == 0:
        return None

    maxima = []
    minima = []
    for point in crit_points:
        for i in range(2, 6):
            ith_deriv = poly.deriv(i)(point)
            # if zero, move up an order of derivative
            if -1.0e-14 < ith_deriv < 1.0e-14:
                continue
            # derivative > 0 and i is even => max
            elif ith_deriv < 0 and i % 2 == 0:
                maxima.append(point)
            # derivative > 0 and i is even => min
            elif ith_deriv > 0 and i % 2 == 0:
                minima.append(point)
            # otherwise inflection point
            else:
                break

    if get_max:
        if len(maxima) == 0:
            return None
        max_vals = [poly(x) for x in maxima]
        return maxima[np.argmax(max_vals)]

    else:
        if len(minima) == 0:
            return None
        min_vals = [poly(x) for x in minima]
        return minima[np.argmin(min_vals)]
