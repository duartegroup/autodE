import numpy as np
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from autode.log import logger
from autode.units import KjMol
from autode.units import KcalMol
from autode.wrappers.ORCA import ORCA
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.XTB import XTB
from autode.config import Config
import os

def plot_2dpes(r1, r2, flat_rel_energy_array, coeff_mat, mep=None, name='2d_scan'):
    """For flat lists of r1, r2 and relative energies plot the PES by interpolating on a 20x20 grid after fitting with
    a 2d polynomial function

    Arguments:
        r1 (np.ndarray): r1 distance points
        r2 (np.ndarray): r2 distance points
        flat_rel_energy_array (np.ndarray): flat array of energies, i.e all energies at r1[0], then r1[1]...
        coeff_mat (np.array): matrix of polynomial coefficients for the energy surface

    Keyword Arguments:
        mep (list(tuple)): list of coordinates on the grid for the min energy pathway across the surface (default: {None})
        name (str): name of the plot (default: {'2d_scan'})
    """
    plt.close()

    file_extension = '.png'

    logger.info(f'Plotting 2D scan and saving to {name}{file_extension}')

    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(r1.min(), r1.max(), nx),
                         np.linspace(r2.min(), r2.max(), ny))
    # polyval2d gives matrix with element i,j = f(x,y) with f being the polynomial defined by m and x = xx[i,j] and y = yy[i,j]
    zz = polynomial.polyval2d(xx, yy, coeff_mat)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    pos1 = ax1.plot_surface(xx, yy, zz, cmap=plt.get_cmap('plasma'), alpha=0.7)
    plt.colorbar(pos1, ax=ax1)
    pos1 = ax1.contour3D(xx, yy, zz, 30, colors='k', antialiased=True)
    if mep is not None:
        mep_r1 = [coord[0] for coord in mep]
        mep_r2 = [coord[1] for coord in mep]
        mep_energies = [polynomial.polyval2d(x, y, coeff_mat) for x, y in mep]
        pos1 = ax1.plot(mep_r1, mep_r2, mep_energies,
                        color='forestgreen', lw=2, alpha=1)
    ax1.view_init(45)
    ax1.set_xlabel('$r1$ / Å')
    ax1.set_ylabel('$r2$ / Å')
    ax1.set_zlabel('∆$E$ / kcal mol$^{-1}$')
    ax2 = fig.add_subplot(1, 2, 2)
    pos2 = ax2.imshow(zz, aspect=(abs(r1.max()-r1.min())/abs(r2.max()-r2.min())), extent=(r1.min(), r1.max(),
                         r2.min(), r2.max()), origin='lower', cmap=plt.get_cmap('plasma'))
    ax2.set_xlabel('$r1$ / Å')
    ax2.set_ylabel('$r2$ / Å')
    plt.colorbar(pos2, ax=ax2)
    if Config.high_qual_plots:
        dpi = 1000
    else:
        dpi = 10
    plt.savefig(name + file_extension, dpi=dpi)


def plot_1dpes(rs, rel_energies, method, scan_name, plot_name='1d_scan'):

    file_extension = Config.image_file_extension

    label = method.__name__

    if 'opt_level' in scan_name:
        label += ' higher level'
        colour = 'deeppink'
    elif 'll1d' in scan_name:
        colour = 'deepskyblue'
    else:
        colour = 'darkmagenta'

    if 'fbond' in scan_name:
        plot_name += '_fbond'
    elif 'bbond' in scan_name:
        plot_name += '_bbond'

    # close the plot so different scan types don't get drawn on top of each other
    if not os.path.exists(plot_name + file_extension):
        plt.close()

    logger.info(f'Plotting 1D scan and saving to {plot_name}{file_extension}')
    plt.plot(rs, rel_energies, marker='o', color=colour, label=label)
    plt.legend()
    plt.xlabel('$r$ / Å')
    plt.ylabel('∆$E$ / kcal mol$^{-1}$')
    if Config.high_qual_plots:
        dpi = 1000
    else:
        dpi = 10
    plt.savefig(plot_name + file_extension, dpi=dpi)


def plot_reaction_profile(e_reac, e_ts, e_prod, units, name, is_true_ts, ts_is_converged):
    """For a reactant reactants -> ts -> products plot the reaction profile using matplotlib

    Arguments:
        e_reac (float): relative reactant energy, usually 0.0
        e_ts (float): relative ts energy
        e_prod (float): relative product energy
        units (object): an object defined in units.py
        name (str): reaction name to annotate to the plot
        is_true_ts (bool): flag for whether the TS is good, i.e. has a single imaginary frequency
        ts_is_converged (bool): flag for whether the TS geometry is converged or not
    """
    logger.info('Plotting reaction profile')

    file_extension = Config.image_file_extension

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
    #make a cubic line from reac to TS, and another from TS to prod
    #reac to TS
    a = np.array([[x**3, x**2], [3*x**2, 2*x]])
    b = np.array([dgdd, 0])
    reac_to_ts = np.linalg.solve(a,b)
    #TS to prod, shift curve so TS at (0,0) to make algebra easier
    y = 1-x
    a = np.array([[y**3, y**2], [3*y**2, 2*y]])
    b = np.array([dg-dgdd, 0])
    ts_to_prod = np.linalg.solve(a,b)

    x_vals = np.linspace(-0.2, 1.2, 140)
    y_vals = []
    begin_x = 0
    end_x = len(x_vals)
    for index, val in enumerate(x_vals):
        if val < x:
            a, b = reac_to_ts
            y = a*val**3 + b*val**2
            #don't want to go up too far at before reacs
            if (val < 0) and not (-1 < y < 1):
                begin_x = index + 1
            else:
                y_vals.append(y + e_reac)
        else:
            a, b = ts_to_prod
            shift_val = val - x
            y = a*shift_val**3 + b*shift_val**2 + dgdd #shift back TS
            if (val > 1) and not ((dg - 1) < y < (dg + 1)):
                end_x = index
                break
            else:
                y_vals.append(y + e_reac)
    
    _, ax = plt.subplots()
    ax.plot(x_vals[begin_x: end_x], y_vals, c='k')

    x_label_coords = [-0.035, x-.035, 0.965]
    x_point_coords = [0, x, 1]
    energies = [np.round(e_reac, 1), np.round(e_ts, 1), np.round(e_prod, 1)]
    for i, energy in enumerate(energies):
        ax.annotate(energy, (x_label_coords[i], energy + 0.05*max(y_vals)), fontsize=12)
        plt.plot(x_point_coords[i], energy, marker='o', markersize=3, color='b')

    if not is_true_ts:
        ax.annotate('TS has >1 imaginary frequency',
                    (0.5, 0.1*max(y_vals)), ha='center', color='red')
    if not ts_is_converged:
        ax.annotate('TS is not fully converged',
                    (0.5, 0.2*max(y_vals)), ha='center', color='red')

    plt.title(name, fontdict={'fontsize': 12})
    plt.xticks([])

    if units == KjMol:
        plt.ylabel('∆$E$ / kJ mol$^{-1}$', fontsize=12)
    if units == KcalMol:
        plt.ylabel('∆$E$/ kcal mol$^{-1}$', fontsize=12)

    plt.ylim(min(y_vals) - 0.05*max(y_vals), 1.2 * max(y_vals))
    if Config.high_qual_plots:
        dpi = 1000
    else:
        dpi = 10
    plt.savefig('reaction_profile' + file_extension, dpi=dpi)


def make_reaction_animation(name, xyzs):
    """makes an xyz file that animates the reaction pathway

    Arguments:
        name (str): name of the xyz file to be created
        xyzs (list of lists): list with each element of the list a list of xyzs
    """
    logger.info('Generating a reaction pathway animation')
    with open(f'{name}_animation.xyz', 'w') as output_file:
        for frame, xyz_list in enumerate(xyzs):
            print(len(xyz_list), file=output_file)
            print(frame, file=output_file)
            [print('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(*line), file=output_file) for line in xyz_list]