from autode.log import logger
import networkx as nx
import numpy as np


def get_mep(r1, r2, energies, saddlepoint):
    """Finds the minimum energy pathway from reactants to products over a given saddlepoint
    
    Arguments:
        r1 {tuple} -- the distances of grid points on one axis
        r2 {tuple} -- the distances of grid points on one axis
        energies {np.array} -- grid of the energy at each grid point given by r1 and r2
        saddlepoint {tuple} -- coordinates of the saddle point in terms of r1 and r2
    
    Returns:
        {list} -- list of grid coordinates corresponding to the mep
    """
    logger.info('Generating minimum energy pathway')
    # reac and prod coords on the grid
    n_points = len(r1)
    reac_coords = (0, 0)
    prod_coords = (n_points-1, n_points-1)

    energies_graph = nx.DiGraph()
    no_negative_weight = np.amax(energies) - np.amin(energies)
    for i in range(n_points):
        for j in range(n_points):
            neighbouring_points = get_neighbouring_points((i, j), n_points)
            for neighbour in neighbouring_points:
                weight = energies[neighbour[0],
                                  neighbour[1]] - energies[i, j] + no_negative_weight
                energies_graph.add_edge((i, j), neighbour, weight=weight)

    # saddlepoint coords on the grid
    saddle_coords = get_point_on_grid(saddlepoint, r1, r2)

    # get two coords so are always just below saddlepoint, and don't have to worry about going over it
    saddle_neighbours = get_neighbouring_points(saddle_coords, n_points)
    saddle_neighbour_energies = [energies[x, y] for x, y in saddle_neighbours]
    lowest_saddle_neighbour_index = saddle_neighbour_energies.index(min(saddle_neighbour_energies))
    one_side_of_saddle = saddle_neighbours[lowest_saddle_neighbour_index]
    other_side_of_saddle = []
    # get opposite point of square around saddle point
    for index, coord in enumerate(one_side_of_saddle):
        if saddle_coords[index] == coord:
            other_side_of_saddle.append(coord)
        elif saddle_coords[index] == coord - 1:
            if not saddle_coords[index] == 0:
                other_side_of_saddle.append(coord - 2)
            else:
                return None
        else:
            if not saddle_coords[index] == n_points - 1:
                other_side_of_saddle.append(coord + 2)
            else:
                return None

    # see which point is closest to reacs
    first_dist_to_reacs = one_side_of_saddle[0] ** 2 + one_side_of_saddle[1]**2
    second_dist_to_reacs = other_side_of_saddle[0] ** 2 + other_side_of_saddle[1]**2
    if first_dist_to_reacs < second_dist_to_reacs:
        reac_saddle_coords = (one_side_of_saddle[0], one_side_of_saddle[1])
        prod_saddle_coords = (other_side_of_saddle[0], other_side_of_saddle[1])
    else:
        reac_saddle_coords = (other_side_of_saddle[0], other_side_of_saddle[1])
        prod_saddle_coords = (one_side_of_saddle[0], one_side_of_saddle[1])

    reac_mep = find_point_on_mep(energies_graph, reac_saddle_coords, reac_coords)
    prod_mep = find_point_on_mep(energies_graph, prod_saddle_coords, prod_coords)

    if saddle_coords in reac_mep or saddle_coords in prod_mep:
        return None

    full_mep = list(reversed(reac_mep)) + [saddle_coords] + prod_mep

    return full_mep


def get_point_on_grid(point, r1, r2):
    """For grid of distances, find the closest point on the grid to a point lying 
    within - but not on - the grid

    Arguments:
        point {tuple} -- the point to be assigned to the grid
        r1 {tuple} -- the distances of grid points on one axis
        r2 {tuple} -- the distances of grid points on one axis

    Returns:
        tuple -- the closest grid point to the point
    """
    r1_diff_from_point = [abs(point[0] - distance) for distance in r1]
    r1_index = r1_diff_from_point.index(min(r1_diff_from_point))
    r2_diff_from_point = [abs(point[1] - distance) for distance in r2]
    r2_index = r2_diff_from_point.index(min(r2_diff_from_point))
    return (r1_index, r2_index)


def find_point_on_mep(energies_graph, start, end):
    """Finds the lowest weighted path from start point to end point

    Arguments:
        energies_graph {nx.Graph} -- graph from a grid of the PES, with 
        the edges having the weight of the energy change between the two nodes
        start {tuple} -- start node
        end {tuple} -- end node

    Returns:
        list -- list of nodes of the minimum energy path
    """
    mep = nx.dijkstra_path(energies_graph, start, end)

    return mep


def get_neighbouring_points(point, n_points):
    """Gets the neigbouring points to a point in a grid

    Arguments:
        point {tuple} -- coords of the point
        n_points {int} -- size of the grid

    Returns:
        list -- list of the neigbouring points
    """
    x, y = point
    if x == 0:
        new_xs = [x, x + 1]
    elif x == n_points-1:
        new_xs = [x - 1, x]
    else:
        new_xs = [x-1, x, x + 1]
    if y == 0:
        new_ys = [y, y + 1]
    elif y == n_points-1:
        new_ys = [y - 1, y]
    else:
        new_ys = [y-1, y, y + 1]
    neighbouring_points = []
    for new_x in new_xs:
        for new_y in new_ys:
            neighbouring_points.append((new_x, new_y))
    neighbouring_points.remove(point)
    return neighbouring_points
