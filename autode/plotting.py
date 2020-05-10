import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from numpy.polynomial import polynomial
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import minimize
from autode.config import Config
from autode.log import logger
from autode.units import KcalMol
from autode.units import KjMol


def save_plot(plot, filename):
    """Save a plot"""

    if os.path.exists(filename):
        logger.warning('Plot already exists. Overriding..')
        os.remove(filename)

    plot.savefig(filename, dpi=800 if Config.high_quality_plots else 100)
    plot.close()

    return None


def plot_2dpes(r1, r2, coeff_mat, mep=None, name='2d_scan'):
    """For flat lists of r1, r2 and relative energies plot the PES by interpolating on a 50x50 grid after fitting with
    a 2d polynomial function

    Arguments:
        r1 (np.ndarray): r1 distance points
        r2 (np.ndarray): r2 distance points
        coeff_mat (np.array): matrix of polynomial coefficients for the energy surface

    Keyword Arguments:
        mep (list(tuple)): list of coordinates on the grid for the min energy pathway across the surface (default: {None})
        name (str): name of the plot (default: {'2d_scan'})
    """
    plt.close()

    try:
        if '_ll2d' in name:
            name_split = name.split('_ll2d')[0].split('_')
        else:
            name_split = name.split('_hl2d')[0].split('_')
        bond1_atoms_ids = [int(idx) for idx in name_split[-2].split('-')]
        bond2_atoms_ids = [int(idx) for idx in name_split[-1].split('-')]
        xlabel = f'$r_1$({bond1_atoms_ids[0]}, {bond1_atoms_ids[1]}) / Å'
        ylabel = f'$r_2$({bond2_atoms_ids[0]}, {bond2_atoms_ids[1]}) / Å'

    except (IndexError, ValueError):
        xlabel = '$r_1$ / Å'
        ylabel = '$r_2$ / Å'

    logger.info(f'Plotting 2D scan and saving to {name}.png')

    nx, ny = 50, 50
    xx, yy = np.meshgrid(np.linspace(r1.min(), r1.max(), nx),
                         np.linspace(r2.min(), r2.max(), ny))

    # polyval2d gives matrix with element i,j = f(x,y) with f being the polynomial defined by m and
    # x = xx[i,j] and y = yy[i,j]

    zz = polynomial.polyval2d(xx, yy, coeff_mat)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(xx, yy, zz, cmap=plt.get_cmap('plasma'))

    ax1.view_init(45)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2 = fig.add_subplot(1, 2, 2)

    pos2 = ax2.imshow(zz, aspect=(abs(r1.max()-r1.min())/abs(r2.max()-r2.min())), extent=(r1.min(), r1.max(),
                                                                                          r2.min(), r2.max()),
                      origin='lower', cmap=plt.get_cmap('plasma'))

    if mep is not None:
        xsmoothed = gaussian_filter1d([coord[0] for coord in mep], sigma=2)
        ysmoothed = gaussian_filter1d([coord[1] for coord in mep], sigma=2)
        ax2.plot(xsmoothed, ysmoothed, color='forestgreen', lw=2)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    cbar = plt.colorbar(pos2, ax=ax2)
    cbar.ax.set_ylabel('∆$E$ / kcal mol$^{-1}$')

    return save_plot(plot=plt, filename=f'{name}.png')


def plot_1dpes(rs, rel_energies, method_name, name='1d_scan'):
    logger.info(f'Plotting 1D scan and saving to {name}.png')

    plt.plot(rs, rel_energies, marker='o', color='k', label=method_name)
    plt.legend()
    plt.xlabel('$r$ / Å')
    plt.ylabel('∆$E$ / kcal mol$^{-1}$')

    return save_plot(plot=plt, filename=f'{name}.png')


def plot_reaction_profile(reactions, units, name):
    """For a set of reactions plot the reaction profile using matplotlib

    Arguments:
        reactions (list(autode.reaction.Reaction):
        units (autode.units.Units):
        name (str):
    """
    logger.info('Plotting reaction profile')

    fig, ax = plt.subplots()

    # Get the energies for the reaction profile (y values) plotted against the reaction coordinate (zi_s)
    energies = calculate_reaction_profile_energies(reactions, units=units)
    zi_s = np.array(range(len(energies)))

    # Minimise a set of spline points so the stationary points have y values given in the energies array
    result = minimize(error_on_stationary_points, x0=energies, args=(energies,), method='BFGS')

    # Use the optimised values to construct a spline function that will be plotted
    optimised_spline = interpolate.CubicSpline(zi_s, result.x, bc_type='clamped')

    # Create more zi values from slightly before the minimum to slightly after the maximum
    fine_zi_s = np.linspace(min(zi_s) - 0.2, max(zi_s) + 0.2, num=500)

    # The new zi values are the stationary points of the optimised function
    zi_s = get_stationary_points(fine_zi_s, optimised_spline.derivative())

    # Plot the function
    ax.scatter(zi_s, optimised_spline(zi_s), c='b')
    ax.plot(fine_zi_s, optimised_spline(fine_zi_s), c='k')

    # Annotate the plot with the relative energies
    for i, energy in enumerate(energies):
        ax.annotate(f'{energy:.1f}', (zi_s[i]-0.05, energy + 0.3), fontsize=12)

    plt.ylabel(f'∆$E$ / {units.name}', fontsize=12)
    plt.ylim(min(energies)-1, max(energies)+1)
    plt.xticks([])
    plt.subplots_adjust(top=0.95, right=0.95)
    fig.text(.1, .05, get_reaction_profile_warnings(reactions), ha='left', fontsize=8, wrap=True)

    return save_plot(plt, filename=f'{name}_reaction_profile.png')


def get_reaction_profile_warnings(reactions):
    """Get a string of warnings for a reaction

    Arguments:
        reactions (list(autode.reaction.Reaction)):
    """
    warnings = ''

    for reaction in reactions:

        if reaction.calc_delta_e() is None:
            warnings += f'∆Er not calculated for {reaction.name}, ∆Er = 0 assumed. '

        if reaction.calc_delta_e_ddagger() is None:
            warnings += f'∆E‡ not calculated for {reaction.name}, barrierless reaction assumed. '

        if reaction.ts is not None:

            n_imag_freqs = len(reaction.ts.imaginary_frequencies)
            if n_imag_freqs != 1:
                warnings += f'TS for {reaction.name} has {n_imag_freqs} imaginary frequencies. '

            if reaction.ts.optts_calc is not None and not reaction.ts.optts_calc.optimisation_converged():
                warnings += f'TS for {reaction.name} was not fully converged. '

    # If no strings were added then there are no warnings
    if len(warnings) == 0:
        warnings = 'None'

    return f'WARNINGS: {warnings}'


def calculate_reaction_profile_energies(reactions, units):
    """Calculate a list of energies comprising the reaction profile

    Arguments:
        reactions (list(autode.reaction.Reaction)):
        units (autode.units.Units):
    """
    # Populate a list of reaction relative energies [reactants -> TS -> products], all floats.
    reaction_energies = []

    for reaction in reactions:

        de = reaction.calc_delta_e()

        # If ∆Er cannot be calculated then assume isoenergetic and add a warning to the plot
        if de is None:
            de = 0.0

        de_ddagger = reaction.calc_delta_e_ddagger()

        # If there is no TS then a barrierless reaction will be assumed and a warning added to the plot
        if de_ddagger is None:
            de_ddagger = 0.0032 + max(0.0, de)

        if not reaction.switched_reacs_prods:
            reaction_energies.append([0.0, de_ddagger, de])

        else:
            # If reactants and products have been switched then correct to the forwards ∆E‡, ∆Er
            reaction_energies.append([0.0, de_ddagger-de, -de])

    # Construct the full list of energies, referenced to the first set of reactants
    energies = reaction_energies[0]

    for i in range(1, len(reaction_energies)):
        # Add the energies from the next TS and the next product reaction_energies[i][0] == energies[-1
        energies += [reaction_energies[i][1] + energies[-1], reaction_energies[i][2] + energies[-1]]

    return units.conversion * np.array(energies)


def get_stationary_points(xs, dydx):
    """
    Compute the productive of the derivative at points x(i-1) and x(i) which is negative if there is a point x(k)
    between x(i-1) and x(i) that has dy/dx|_x(k) = 0

    Arguments:
         xs (np.ndarray):
         dydx (function):
    """
    stationary_points = []

    for i in range(1, len(xs) - 1):

        if dydx(xs[i - 1]) * dydx(xs[i]) < 0:
            stationary_points.append(xs[i])

    return stationary_points


def error_on_stationary_points(x, energies):
    """
    Calculate the difference between the stationary points of an interpolated function and those observed
    (given in the energies array)

      |     .
    E |.   / \        The points indicate the true stationary points
      | \_/   \.
      |_____________
            zi

    Arguments:
        x (np.ndarray): Points that will be splined that generate stationary points that ≈ energies
        energies (np.ndarray): Observed stationary points

    Returns:
        (float): A measure of the error
    """
    # Generate a list of reaction coordinate points - arbitrary units so integers are fine
    zi_s = np.array(range(len(x)))

    # Spline the energies to get a function that has stationary points
    spline = interpolate.CubicSpline(zi_s, x, bc_type='clamped')

    # Calculate the energy values at the stationary points of the function with a fine-ish spacing that extrapolates
    # slightly
    fine_zi_s = np.linspace(min(zi_s)-0.2, max(zi_s)+0.2, num=100)
    stationary_points = get_stationary_points(xs=fine_zi_s, dydx=spline.derivative())
    energies_at_stationary_points = [spline(zi) for zi in stationary_points]

    # Return the error as the sum squared difference between the required and the observed stationary point energies
    energy_difference = energies - np.array(energies_at_stationary_points)
    return np.sum(np.square(energy_difference))

