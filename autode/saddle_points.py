from scipy.optimize import minimize, Bounds
from autode.log import logger
import numpy as np


def poly2d_saddlepoints(coeff_mat, xs, ys):
    """Finds the saddle points of a 2d surface defined by a matrix of coefficients

    Arguments:
        coeff_mat (np.array): Matrix of coefficients of the n order polynomial
        xs (float) (np.ndarray): 1D
        ys (float) (np.ndarray): 1D

    Returns:
        list: list of saddle points
    """
    logger.info('Finding saddle points')
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    stationary_points = []

    # Calculate the derivatives over a uniform grid in x, y. 10x10 should find all the unique saddle points
    for x in np.linspace(min_x, max_x, num=10):
        for y in np.linspace(min_y, max_y, num=10):

            # Minimise (df/dx)^2 + (dy/dx)^2 with bounds ensuring the saddle points are within the surface
            opt = minimize(sum_squared_xy_derivative,
                           x0=np.array([x, y]), args=(coeff_mat,),
                           method='TNC',
                           bounds=Bounds(lb=np.array([min_x, min_y]),
                                         ub=np.array([max_x, max_y])))
            opt_x, opt_y = opt.x

            # Check that we're still inside the bounds and the optimisation has converged reasonably
            if min_x < opt_x < max_x and min_y < opt_y < max_y and opt.fun < 1E-1:
                stationary_points.append(opt.x)

    # Remove all repeated stationary points
    stationary_points = get_unique_stationary_points(stationary_points)

    # Return all stationary points that are first order saddle points (i.e. could be a TS)
    saddle_points = [point for point in stationary_points if is_saddle_point(point, coeff_mat)]
    logger.info(f'Found {len(saddle_points)} saddle points')

    saddle_points = get_sorted_saddlepoints(saddle_points=saddle_points, xs=xs, ys=ys)
    return saddle_points


def get_sorted_saddlepoints(saddle_points, xs, ys):
    """Get the list of saddle points ordered by their distance from the (x, y) mid-point"""

    mid_x, mid_y = np.average(xs), np.average(ys)

    return sorted(saddle_points, key=lambda point: np.abs(point[0] - mid_x) + np.abs(point[1] - mid_y))


def get_unique_stationary_points(stationary_points, dist_threshold=0.1):
    """Strip all points that are close to each other"""
    logger.info(f'Have {len(stationary_points)} stationary points')

    unique_stationary_points = stationary_points[:1]

    for stat_point in stationary_points[1:]:

        # Assume the point in unique and determine if it is close to any of the point already in the list
        unique = True

        for unique_stat_point in unique_stationary_points:
            distance = np.sqrt(np.sum(np.square(np.array(stat_point) - np.array(unique_stat_point))))
            if distance < dist_threshold:
                unique = False

        if unique:
            unique_stationary_points.append(stat_point)

    logger.info(f'Stripped {len(stationary_points) - len(unique_stationary_points)} stationary points')
    return unique_stationary_points


def sum_squared_xy_derivative(xy_point, coeff_mat):
    """For a coordinate, and function, finds df/dx and df/dy and returns the sum of the squares

    Arguments:
        xy_point (tuple): (x,y)
        coeff_mat (np.array): Matrix of coefficients of the n order polynomial

    Returns:
        (float): (df/dx + df/dy)^2 where at a stationary point ~ 0
    """
    order = coeff_mat.shape[0]
    x, y = xy_point
    dx, dy = 0, 0

    for i in range(order):  # x index
        for j in range(order):  # y index
            if i > 0:
                dx += coeff_mat[i, j] * i * x**(i-1) * y**j
            if j > 0:
                dy += coeff_mat[i, j] * x**i * j * y**(j-1)

    return dx**2 + dy**2


def is_saddle_point(xy_point, coeff_mat):
    """
    Calculates whether a point (x, y) is a saddle point by computing

    delta = ((d2f/dx2)*(d2f/dy2) - (d2f/dxdy)**2)

    Arguments:
        coeff_mat (np.array): Matrix of the coefficients of the n order polynomial (n x n)
        xy_point (tuple): the stationary point to be examined

    Returns:
         (bool):
    """
    dx2, dy2, dxdy = 0, 0, 0
    x, y = xy_point

    order = coeff_mat.shape[0]
    for i in range(order):  # x index
        for j in range(order):  # y index
            if i > 1:
                dx2 += coeff_mat[i, j] * i * (i - 1) * x**(i - 2) * y**j
            if j > 1:
                dy2 += coeff_mat[i, j] * x**i * j * (j - 1) * y**(j - 2)
            if i > 0 and j > 0:
                dxdy += coeff_mat[i, j] * i * x**(i - 1) * j * y**(j - 1)

    if dx2 * dy2 - dxdy**2 < 0:
        logger.info(f'Found saddle point at r1 = {x:.3f}, r2 = {y:.3f} Å')
        return True

    else:
        return False
