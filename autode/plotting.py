import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial import polynomial
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d
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
        name_split = name.split('_')
        bond1 = name_split[-3]
        bond2 = name_split[-2]
        bond1_atoms = [int(atom) for atom in bond1.split('-')]
        bond2_atoms = [int(atom) for atom in bond2.split('-')]
        xlabel = f'$r_1$({bond1_atoms[0]}, {bond1_atoms[1]}) / Å'
        ylabel = f'$r_2$({bond2_atoms[0]}, {bond2_atoms[1]}) / Å'

    except IndexError:
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


def plot_reaction_profile(e_reac, e_ts, e_prod, units, reacs, prods, ts, switched=False):
    """For a reactant reactants -> ts -> products plot the reaction profile using matplotlib

    Arguments:
        e_reac (float): Relative reactant energy, usually 0.0
        e_ts (float): Relative ts energy or None
        e_prod (float): Relative product energy
        units (autode.units.Units):
        reacs (list(autode.molecule.Reactant)):
        prods (list(autode.molecule.Product)):
        ts (autode.transtion_states.transtion_state.TranstionState):
        switched (bool): flag for a reaction that was initially reversed reactant/products
    """
    logger.info('Plotting reaction profile')

    if e_ts is None:
        logger.error('TS is None – assuming barrierless reaction. ∆E‡ = 2 kcal mol-1 above the maximum energy')
        e_ts = 0.0032 * units.conversion + max(0.0, e_prod)

    if switched:
        # Swap the energies of reactants and products
        e_reac, e_prod = e_prod, e_reac
        e_ts -= e_reac
        e_prod -= e_reac
        e_reac = 0.0
        reacs, prods = prods, reacs

    # Define the plot name
    name = ' + '.join([r.name for r in reacs]) + ' → ' + ' + '.join([p.name for p in prods])

    dg = e_prod - e_reac
    dgdd = e_ts - e_reac

    # 0 < x < 1, position of TS along reaction coordinate
    if dg > 0:
        if dgdd < dg:
            x = dgdd/dg
        else:
            x = 1 - (dgdd - dg)/(2*dgdd)
    else:
        if dgdd < 0:
            x = dgdd/dg
        else:
            x = (dgdd)/(2*(dgdd-dg))

    # make a cubic line from reac to TS, and another from TS to prod
    # reac to TS
    a = np.array([[x**3, x**2], [3*x**2, 2*x]])
    b = np.array([dgdd, 0])
    reac_to_ts = np.linalg.solve(a, b)

    # TS to prod, shift curve so TS at (0,0) to make algebra easier
    y = 1-x
    a = np.array([[y**3, y**2], [3*y**2, 2*y]])
    b = np.array([dg-dgdd, 0])
    ts_to_prod = np.linalg.solve(a, b)

    x_vals = np.linspace(-0.2, 1.2, 140)
    y_vals = []
    begin_x = 0
    end_x = len(x_vals)
    for index, val in enumerate(x_vals):
        if val < x:
            a, b = reac_to_ts
            y = a*val**3 + b*val**2

            # Don't want to go up too far at before reacs
            if (val < 0) and not (-1 < y < 1):
                begin_x = index + 1
            else:
                y_vals.append(y + e_reac)
        else:
            a, b = ts_to_prod
            shift_val = val - x
            y = a*shift_val**3 + b*shift_val**2 + dgdd  # shift back TS
            if (val > 1) and not ((dg - 1) < y < (dg + 1)):
                end_x = index
                break
            else:
                y_vals.append(y + e_reac)

    _, ax = plt.subplots()
    ax.plot(x_vals[begin_x: end_x], y_vals, c='k')

    y_range = max(y_vals) - min(y_vals)

    x_label_coords = [-0.035, x-.035, 0.965]
    y_label_shift = [0.04*y_range, -0.07*y_range, 0.04*y_range]
    x_point_coords = [0, x, 1]
    energies = [np.round(e_reac, 1), np.round(e_ts, 1), np.round(e_prod, 1)]

    for i, energy in enumerate(energies):
        if ts is None and i == 1:
            continue
        ax.annotate(energy, (x_label_coords[i], energy + y_label_shift[i]), fontsize=12)
        plt.plot(x_point_coords[i], energy, marker='o', markersize=3, color='b')

    if ts is not None:
        if len(ts.imaginary_frequencies) != 1:
            ax.annotate(f'TS has {len(ts.imaginary_frequencies)} imaginary frequency',
                        (0.5, 0.1*max(y_vals)), ha='center', color='red')

        if ts.optts_calc is not None and not ts.optts_calc.optimisation_converged():
            ax.annotate('TS is not fully converged',
                        (0.5, 0.2*max(y_vals)), ha='center', color='red')
    else:
        ax.annotate('No TS was found, barrierless reaction assumed', (0.5, 0.3*max(y_vals)), ha='center', color='red')

    plt.title(name, fontdict={'fontsize': 12})
    plt.xticks([])

    if units == KjMol:
        plt.ylabel('∆$E$ / kJ mol$^{-1}$', fontsize=12)
    if units == KcalMol:
        plt.ylabel('∆$E$/ kcal mol$^{-1}$', fontsize=12)

    plt.ylim(min(y_vals) - 0.1*max(y_vals), 1.2 * max(y_vals))

    return save_plot(plt, filename='reaction_profile.png')
