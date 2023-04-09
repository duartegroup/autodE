"""
Routines for polynomial fitted line searches used in
optimisers: quintic, quartic and cubic fits are implemented

The result of the line searches are represented as fractions
of the distance between the two points (0 and 1), i.e. if the
result is 0.5, the minimum is halfway along the line 0->1; if
the result is 1.2 the minimum is 1.2 times the 0>1 vector or
0.2 * 0->1 on the side of point 1. If the result is in [0,1]
it is an interpolation, otherwise (e.g. > 1) extrapolation
"""
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
from numpy.polynomial import Polynomial
import scipy
from autode.exceptions import OptimiserStepError, CoordinateTransformFailed
from autode.log import logger

if TYPE_CHECKING:
    from autode.opt.coordinates import OptCoordinates

# range within which to search for minima (quintic polynomial can have multiple minima)
_poly_minim_search_range = [0, 4]


def polynomial_line_search(
    coord0: OptCoordinates, coord1: OptCoordinates, use_quintic: bool = False
) -> Optional[OptCoordinates]:
    """
    Polynomial fitted 1D line search using two coordinates, requires
    the energies and gradients for cubic and constrained quartic
    line search, and hessian for quintic search.

    Args:
        coord0: Previous coordinate point
        coord1: Current coordinate point
        use_quintic: whether to use Hessian information for quintic fit

    Returns:
        (OptCoordinates): Final coordinate, containing the energy and
                          gradient from polynomial fit
    """
    coord1 = coord1.copy()
    coord1.allow_unconverged_back_transform = False
    # generate directional gradients
    assert coord0.e is not None and coord0.g is not None
    assert coord1.e is not None and coord1.g is not None

    e0 = coord0.e
    e1 = coord1.e
    # if latest point is not lowest in energy, must not extrapolate
    if e1 > e0:
        upper_limit = 1
    else:
        upper_limit = _poly_minim_search_range[1]
    step = coord1.raw - coord0.raw  # line along 0->1
    g0 = float(np.dot(step, coord0.g))
    g1 = float(np.dot(step, coord1.g))

    def calculate_new_coord(new_x, new_e) -> Tuple[OptCoordinates, float]:
        """
        Generate new coordinate and gradient from the fraction representing
        position of the minimum. new_x is the fractional position and new_e
        is the interpolated energy
        """
        if new_e > coord1.e:
            raise OptimiserStepError("Unknown error in polynomial fitting")
        if new_x > upper_limit:
            raise OptimiserStepError("Polynomial interpolation too large")
        fitted_step = (new_x - 1) * step
        fitted_grad = (1 - new_x) * coord0.g + new_x * coord1.g
        fitted_coord = coord1 + fitted_step
        fitted_coord.e = new_e
        fitted_coord.g = fitted_grad
        step_size = np.linalg.norm(fitted_coord.to("cart") - coord1.to("cart"))
        return fitted_coord, step_size

    try:
        if not use_quintic:
            raise ValueError
        # try quintic interpolation only if requested, and hessian available
        assert coord0.h is not None and coord1.h is not None
        h0 = float(np.linalg.multi_dot((step, coord0.h, step)))
        h1 = float(np.linalg.multi_dot((step, coord1.h, step)))
        x, en = quintic_fit_get_minimum(e0, g0, h0, e1, g1, h1)
        if x is None:
            raise OptimiserStepError("Quintic interpolation failed")
        new_coord, size = calculate_new_coord(x, en)
        logger.info(
            f"Quintic line search step: {size} Å, predicted E = {en:.6f}"
        )
        return new_coord

    except (ValueError, OptimiserStepError, CoordinateTransformFailed):
        pass

    try:
        x, en = constrained_quartic_fit_get_minimum(e0, g0, e1, g1)
        if x is None:
            raise OptimiserStepError("Quartic interpolation failed")
        new_coord, size = calculate_new_coord(x, en)
        logger.info(
            f"Quartic line search step: {size} Å, predicted E = {en:.6f}"
        )
        return new_coord

    except (OptimiserStepError, CoordinateTransformFailed):
        pass

    try:
        x, en = cubic_fit_get_minimum(e0, g0, e1, g1)
        # hard limit on cubic interpolation: step size should be lower than last step
        if x is None or x > 2:
            raise OptimiserStepError("Cubic interpolation failed")
        new_coord, size = calculate_new_coord(x, en)
        logger.info(
            f"Cubic line search step: {size} Å, predicted E = {en:.6f}"
        )
        return new_coord
    except (OptimiserStepError, CoordinateTransformFailed):
        pass

    return None


def quintic_fit_get_minimum(
    e0: float, g0: float, h0: float, e1: float, g1: float, h1: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    A linear search (1D) based on two points, by fitting a fifth
    order (quintic) polynomial to the energies, and projected gradients
    and hessians along the line between the points, to obtain the minimum
    in that direction.

    Args:
        e0: energy at first point
        g0: projected gradient(1D) at first point
        h0: projected Hessian(1D) at first point
        e1: energy at second point
        g1: projected gradient(1D) at second point
        h1: projected Hessian(1D) at second point

    Returns:
        (tuple): The minimum point, and the predicted energy
    """
    # quintic: f(x) = a + bx + cx^2 + dx^3 + px^4 + qx^5
    # f'(x) = b + 2 c x + 3 d x^2 + 4 p x^3 + 5 q x^4
    # f''(x) = 2 c + 6 d x + 12 p x^2 + 20 q x^3
    # e0 = f(0) = a; g0 = f'(0) = b; h0 = f''(0) = 2 c
    a, b, c = e0, g0, h0 / 2
    # e1 = f(1) = a + b + c + d + p + q => d + p + q = e1 - (a+b+c)
    # g1 = f'(1) = b + 2c + 3d + 4p + 5q => 3d + 4p + 5q = g1 - (b+2c)
    # h1 = f''(1) = 2c + 6d + 12p + 20q => 6d + 12p + 20q = h1 - 2c
    coeff_mat = np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float)
    b_mat = np.array([e1 - (a + b + c), g1 - (b + 2 * c), h1 - 2 * c])
    try:
        x = scipy.linalg.solve(coeff_mat, b_mat)
    except scipy.linalg.LinAlgError:
        return None, None

    assert x.shape == (3,)
    d, p, q = list(x)
    quint_poly = Polynomial([a, b, c, d, p, q])
    return _get_poly_minimum(quint_poly)


def constrained_quartic_fit_get_minimum(
    e0: float, g0: float, e1: float, g1: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    A linear search (1D) based on two points fitted to a constrained
    fourth order (quartic) polynomial, to obtain the minimum along
    that line. The constraint is that the second derivative of the
    polynomial has to be 0 at only one point. The result is a fraction
    denoting the position of the minimum on the line, represented
    relative to the two points.

    Args:
        e0: energy at first point
        g0: projected gradient (1D) at first point
        e1: energy at second point
        g1: projected gradient (1D) at second point

    Returns:
        (tuple): The minimum point, and the predicted energy
    """
    # quartic: f(x) = a + bx + cx^2 + dx^3 + px^4
    # f'(x) = b + 2 c x + 3 d x^2 + 4 p x^3
    # e0 = f(0) = a; g0 = f'(0) = b
    a, b = e0, g0
    # from sympy result
    square_term = (
        6 * (e0 - e1) ** 2
        + 6 * (e0 - e1) * (g0 + g1)
        + (g0 + g1) ** 2
        + 2 * g0 * g1
    )
    if square_term < 1.0e-15:
        return None, None
    sym1 = np.sqrt(square_term)

    # components of a2, a3 that appear in all expressions
    c_comp = -3 * (e0 - e1) - 5 * g0 / 2 - g1 / 2
    d_comp = 2 * e0 - 2 * e1 + 2 * g0

    # first solution
    c = c_comp - sym1 / 2
    d = d_comp + sym1
    p = (3 / 8) * d**2 / c
    quartic_poly_first = Polynomial([a, b, c, d, p])

    # second solution
    c = c_comp + sym1 / 2
    d = d_comp - sym1
    p = (3 / 8) * d**2 / c
    quartic_poly_second = Polynomial([a, b, c, d, p])

    # get the minimum
    minim_first = _get_poly_minimum(quartic_poly_first)
    minim_second = _get_poly_minimum(quartic_poly_second)

    # both solutions should have minima, choose the one with lowest energy
    assert minim_first[0] is not None and minim_second[0] is not None
    if minim_first[1] < minim_second[1]:
        return minim_first
    else:
        return minim_second


def cubic_fit_get_minimum(
    e0: float, g0: float, e1: float, g1: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    A linear search based on two points fitted to a cubic polynomial,
    to obtain the minimum along that line.

    Args:
        e0: energy at first point
        g0: projected gradient (1D) at first point
        e1: energy at second point
        g1: projected gradient (1D) at second point

    Returns:
        (tuple): The minimum point, and the predicted energy
    """
    # cubic: f(x) = a + b x + c x^2 + d x^3
    # f'(x) = b + 2 c x + 3 d x^2

    # e0 = f(0) = a; g0 = f'(0) = b
    a, b = e0, g0
    # e1 = f(1) = a + b + c + d => c + d = e1 - a - b
    c_d = e1 - a - b
    # g1 = f'(1) = b + 2 c + 3 d => 2c + 3d = g1 - b
    c2_d3 = g1 - b
    # d = 2 c + 3 d - 2 (c + d)
    d = c2_d3 - 2 * c_d
    c = c_d - d
    cubic_poly = Polynomial([a, b, c, d])
    return _get_poly_minimum(cubic_poly)


def _get_poly_minimum(
    poly: Polynomial,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns the minimum of a polynomial, or None if there is
    no minima. Uses derivative test for analytic derivation
    of the minima.

    Args:
        poly (Polynomial): The numpy polynomial object

    Returns:
        (tuple): the x value for which f(x) is the minimum,
                 and the predicted value (energy) at that point
    """
    critical_points = poly.deriv(1).roots()
    real_crit_points = critical_points[critical_points.imag == 0].real
    if len(real_crit_points) == 0:
        return None, None

    minima = []
    for point in real_crit_points:
        for i in range(2, 6):
            i_th_deriv = poly.deriv(i)(point)
            if -1.0e-12 < i_th_deriv < 1.0e-12:
                # zero deriv, go up another order
                continue
            elif i % 2 != 0:
                # odd order derivative, inflection point
                break
            elif i_th_deriv > 1.0e-12:
                # strict minimum
                minima.append(point)
            else:
                # strict maximum
                break

    # get everything in the range [0, 2]
    minima = np.ndarray(minima)
    u_bound, l_bound = _poly_minim_search_range
    minima = minima[(minima >= l_bound) & (minima <= u_bound)]

    if len(minima) == 0:
        return None, None
    values = poly(minima)
    global_minimum = minima[np.argmin(values)]
    min_value = values[np.argmin(values)]
    return global_minimum, min_value
