from scipy import optimize
from autode.log import logger
from autode.min_energy_pathway import get_mep
from autode.min_energy_pathway import get_point_on_grid


def poly2d_saddlepoints(coeff_mat):
    """Finds the saddle points of a 2d surface defined by a matrix of coefficients

    Arguments:
        coeff_mat {np.array} -- Matrix of coefficients of the n order polynomial

    Returns:
        list -- list of saddle points
    """

    logger.info('Finding saddle points')
    stationary_points = []
    # start in every place on the surface to ensure all relevant stationary points are found
    for i in [n/10 for n in range(15, 35)]:
        for j in [m/10 for m in range(15, 35)]:
            sol = optimize.root(root_finder, [i, j], args=(coeff_mat))
            stationary_points.append(sol.x.tolist())

    # sometimes finds extra stationary points, so remove them now
    true_stationary_points = []
    for stationary_point in stationary_points:
        dx, dy = root_finder(stationary_point, coeff_mat)
        if (-0.00001 < dx < 0.00001) and (-0.00001 < dy < 0.00001):
            true_stationary_points.append(stationary_point)

    # remove repeats
    unique_stationary_points = []
    for stationary_point in true_stationary_points:
        unique = True
        for unique_point in unique_stationary_points:
            x_same = False
            y_same = False
            if unique_point[0] - 0.1 < stationary_point[0] < unique_point[0] + 0.1:
                x_same = True
            if unique_point[1] - 0.1 < stationary_point[1] < unique_point[1] + 0.1:
                y_same = True
            if x_same and y_same:
                unique = False
                break
        if unique:
            unique_stationary_points.append(stationary_point)

    # now see which stationary points are saddle points
    saddle_points = []
    for stationary_point in unique_stationary_points:
        if calc_delta(coeff_mat, stationary_point) < 0:
            saddle_points.append(stationary_point)

    return saddle_points


def root_finder(vector, coeff_mat):
    """For a coordinate, and function, finds df/dx and df/dy

    Arguments:
        vector {tuple} -- (x,y)
        coeff_mat {np.array} -- Matrix of coefficients of the n order polynomial

    Returns:
        tuple -- (df/dx, df/dy)
    """
    order = coeff_mat.shape[0]
    x, y = vector
    dx = 0
    dy = 0
    for i in range(order):  # x index
        for j in range(order):  # y index
            if i > 0:
                dx += coeff_mat[i][j] * i * x**(i-1) * y**(j)
            if j > 0:
                dy += coeff_mat[i][j] * x**(i) * j * y**(j-1)
    return dx, dy


def calc_delta(coeff_mat, root):
    """calculates delta ((d2f/dx2)*(d2f/dy2) - (d2f/dxdy)**2), to determine if the stationary point is a saddle point (delta < 0) 

    Arguments:
        coeff_mat {np.array} -- Matrix of the coefficients of the n order polynomial
        root {tuple} -- the stationary point to be examined

    Returns:
        delta {int} -- value of delta
    """
    dx2 = 0
    dy2 = 0
    dxdy = 0
    x, y = root
    order = coeff_mat.shape[0]
    for i in range(order):  # x index
        for j in range(order):  # y index
            if i > 1:
                dx2 += coeff_mat[i][j] * i * (i-1) * x**(i-2) * y**(j)
            if j > 1:
                dy2 += coeff_mat[i][j] * x**(i) * j * (j-1) * y**(j-2)
            if i > 0 and j > 0:
                dxdy += coeff_mat[i][j] * i * x**(i-1) * j * y**(j-1)
    delta = dx2 * dy2 - dxdy**2
    return delta


def best_saddlepoint(saddle_points, r1, r2, energy_grid):
    saddle_points_on_mep = []
    min_energy_pathways = []

    for saddle_point in saddle_points:
        min_energy_pathway = get_mep(r1, r2, energy_grid, saddle_point)
        if min_energy_pathway is not None:
            saddle_points_on_mep.append(saddle_point)
            min_energy_pathways.append(min_energy_pathway)

    if len(saddle_points_on_mep) == 0:
        logger.error(
            'No saddle points were found on the minimum energy pathway')
        return None
    elif len(saddle_points_on_mep) == 1:
        min_energy_pathway = min_energy_pathways[0]
        r1_saddle, r2_saddle = saddle_points_on_mep[0]
    elif len(saddle_points_on_mep) > 1:
        logger.warning(
            'Multiple saddlepoints remain, choosing the highest peak on the lowest minimum energy pathway')

        peak_of_meps = []
        for mep in min_energy_pathways:
            energy_of_mep = [energy_grid[x, y] for x, y in mep]
            peak_of_meps.append(max(energy_of_mep))
        lowest_mep_index = peak_of_meps.index(min(peak_of_meps))
        min_energy_pathway = min_energy_pathways[lowest_mep_index]
        grid_saddlepoints_in_lowest_mep = []
        saddlepoints_in_lowest_mep = []

        for saddlepoint in saddle_points_on_mep:
            saddlepoint_on_grid = get_point_on_grid(saddlepoint, r1, r2)
            if saddlepoint_on_grid in min_energy_pathway:
                grid_saddlepoints_in_lowest_mep.append(saddlepoint_on_grid)
                saddlepoints_in_lowest_mep.append(saddlepoint)

        saddlepoint_in_lowest_mep_energies = [
            energy_grid[x, y] for x, y in grid_saddlepoints_in_lowest_mep]
        max_saddle_energy_index = saddlepoint_in_lowest_mep_energies.index(
            max(saddlepoint_in_lowest_mep_energies))
        r1_saddle, r2_saddle = saddlepoints_in_lowest_mep[max_saddle_energy_index]

    return r1_saddle, r2_saddle, min_energy_pathway
