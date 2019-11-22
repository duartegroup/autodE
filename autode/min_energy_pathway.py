from autode.log import logger


def get_mep(r1, r2, energies, saddlepoint):
    logger.info('Generating minimum energy pathway')
    # reac and prod coords on the grid
    n_points = len(r1)
    reac_coords = (0, 0)
    prod_coords = (n_points-1, n_points-1)

    # saddlepoint coords on the grid
    saddle_coords = get_point_on_grid(saddlepoint, r1, r2)

    # get two coords so are always just below saddlepoint, and don't have to worry about going over it
    saddle_neighbours = get_neighbouring_points(saddle_coords, n_points)
    saddle_neighbour_energies = [energies[x, y] for x, y in saddle_neighbours]
    lowest_saddle_neighbour_index = saddle_neighbour_energies.index(
        min(saddle_neighbour_energies))
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
                other_side_of_saddle.append(0)
        else:
            if not saddle_coords[index] == n_points - 1:
                other_side_of_saddle.append(coord + 2)
            else:
                other_side_of_saddle.append(n_points - 1)

    # see which point is closest to reacs
    first_dist_to_reacs = one_side_of_saddle[0] ** 2 + one_side_of_saddle[1]**2
    second_dist_to_reacs = other_side_of_saddle[0] ** 2 + \
        other_side_of_saddle[1]**2
    if first_dist_to_reacs < second_dist_to_reacs:
        reac_saddle_coords = (one_side_of_saddle[0], one_side_of_saddle[1])
        prod_saddle_coords = (other_side_of_saddle[0], other_side_of_saddle[1])
    else:
        reac_saddle_coords = (other_side_of_saddle[0], other_side_of_saddle[1])
        prod_saddle_coords = (one_side_of_saddle[0], one_side_of_saddle[1])

    reac_mep = find_point_on_mep(
        energies, reac_coords, reac_saddle_coords, n_points, saddle_coords)
    prod_mep = find_point_on_mep(
        energies, prod_coords, prod_saddle_coords, n_points, saddle_coords)

    if reac_mep is None or prod_mep is None:
        return None

    full_mep = list(reversed(reac_mep)) + [saddle_coords] + prod_mep

    return full_mep


def get_point_on_grid(point, r1, r2):
    r1_diff_from_point = [abs(point[0] - distance) for distance in r1]
    r1_index = r1_diff_from_point.index(min(r1_diff_from_point))
    r2_diff_from_point = [abs(point[1] - distance) for distance in r2]
    r2_index = r2_diff_from_point.index(min(r2_diff_from_point))
    return (r1_index, r2_index)


def find_point_on_mep(energies, point_to_find, start, n_points, other_side):
    # fit sometimes leaves the end not at a minimum, so check for this
    end_neighbours = get_neighbouring_points(point_to_find, n_points)
    end_neighbouring_energies = [energies[x, y] for x, y in end_neighbours]
    path_to_point_from_min = []
    if any([energy < energies[point_to_find[0], point_to_find[1]] for energy in end_neighbouring_energies]):
        path_to_point_from_min.append(point_to_find)
        going_down = True
        while going_down:
            current_point = path_to_point_from_min[-1]
            point_energy = energies[current_point[0], [current_point[1]]]
            neighbouring_points = get_neighbouring_points(
                current_point, n_points)
            neighbouring_energies = [energies[x, y]
                                     for x, y in neighbouring_points]
            if any([energy < point_energy for energy in neighbouring_energies]):
                min_energy_index = neighbouring_energies.index(
                    min(neighbouring_energies))
                min_coord = neighbouring_points[min_energy_index]
                path_to_point_from_min.append(min_coord)
            else:
                going_down = False
                point_to_find = current_point
                path_to_point_from_min.remove(current_point)

    found_point = False
    mep = [other_side, start]
    while not found_point:
        current_point = mep[-1]
        neighbouring_points = get_neighbouring_points(current_point, n_points)
        new_points = [
            point for point in neighbouring_points if not point in mep]
        if len(new_points) == 0:
            return None
        neighbouring_energies = [energies[x, y] for x, y in new_points]
        min_energy_index = neighbouring_energies.index(
            min(neighbouring_energies))
        min_coord = new_points[min_energy_index]
        mep.append(min_coord)
        if min_coord == point_to_find:
            found_point = True
    return mep[1:] + list(reversed(path_to_point_from_min))


def get_neighbouring_points(point, n_points):
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
