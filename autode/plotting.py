import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from autode.log import logger
from autode.units import KjMol
from autode.units import KcalMol
from autode.wrappers.ORCA import ORCA
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.XTB import XTB
import os


def plot_2dpes(r1, r2, flat_rel_energy_array, name='2d_scan'):
    """
    For flat lists of r1, r2 and relative energies plot the PES by interpolating on a 20x20 grid after fitting with
    a 2d polynomial function
    :param r1: (np.ndarray)
    :param r2: (np.ndarray)
    :param flat_rel_energy_array: (np.ndarray)
    :param name (str)
    :return:
    """

    def polyval2d(x, y, c):
        # order = int(np.sqrt(len(m))) - 1
        # ij = itertools.product(range(order + 1), range(order + 1)))
        ij = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2)]

        z = np.zeros_like(x)
        for a, (i, j) in zip(c, ij):
            z += a * x ** i * y ** j
        return z

    def polyfit2d(x, y, z):  # order=2
        logger.info('Fitting 2D surface to 2nd order polynomial in x and y')
        # ncols = (order + 1) ** 2
        ij = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2)]
        g = np.zeros((x.size, len(ij)))
        # ij = itertools.product(range(order + 1), range(order + 1)))
        for k, (i, j) in enumerate(ij):
            # print(k, 'x order', i, 'y order', j)
            g[:, k] = x ** i * y ** j
        c, _, _, _ = np.linalg.lstsq(g, z, rcond=None)
        return c

    r1_flat, r2_flat = r1.flatten(), r2.flatten()
    m = polyfit2d(r1_flat, r2_flat, flat_rel_energy_array)
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(r1.min(), r1.max(), nx),
                         np.linspace(r2.min(), r2.max(), ny))
    zz = polyval2d(xx, yy, m)

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    pos1 = ax1.plot_surface(xx, yy, zz, cmap=plt.get_cmap('plasma'))
    ax2 = fig.add_subplot(1, 2, 2)
    pos2 = ax2.imshow(zz, extent=(r1.min(), r2.max(), r1.min(
    ), r2.max()), origin='lower', cmap=plt.get_cmap('plasma'))
    # plt.scatter(r1_flat, r2_flat, c=flat_rel_energy_array)
    plt.colorbar(pos1, ax=ax1)
    plt.colorbar(pos2, ax=ax2)
    plt.savefig(name + '.png')

    return 0


def plot_1dpes(rs, rel_energies, method, scan_name, plot_name='1d_scan'):

    colour_method_dict = {'orca pbe': 'darkmagenta', 'orca pbe0': 'deeppink',
                          'xtb': 'deepskyblue', 'mopac': 'forestgreen'}

    label = method.__name__

    if label.lower() == 'orca':
        label += ' PBE'

    if 'opt_level' in scan_name:
        label += '0'

    colour = colour_method_dict.get(label.lower(), 'black')

    if 'fbond' in scan_name:
        plot_name += '_fbond'
    elif 'bbond' in scan_name:
        plot_name += '_bbond'

    # close the plot so different scan types don't get drawn on top of each other
    if not os.path.exists(plot_name + '.png'):
        plt.close()

    logger.info(f'Plotting 1D scan and saving to {plot_name}.png')
    plt.plot(rs, rel_energies, marker='o', color=colour, label=label)
    plt.legend()
    plt.xlabel('$r$ / Å')
    plt.ylabel('∆$E$ / kcal mol$^{-1}$')
    plt.savefig(plot_name + '.png')

    return 0


def plot_reaction_profile(e_reac, e_ts, e_prod, units, name, is_true_ts, ts_is_converged):
    """
    For a reactant reactants -> ts -> products plot the reaction profile using matplotlib
    :param e_reac: (float) relative reactant energy, usually 0.0
    :param e_ts: (float)
    :param e_prod: (float)
    :param units: (object) an object defined in units.py
    :param name: (str) reaction name to annotate to the plot
    :param is_true_ts: (bool) flag for whether the TS is good, i.e. has a single imaginary frequency
    :param ts_is_converged: (bool) flag for whether the TS geometry is converged or not
    :return:
    """
    logger.info('Plotting reaction profile')
    marker_width = 0.2

    xs = [0.05, 1.0, 1.86]
    ys = [np.round(e_reac, 1), np.round(e_ts, 1), np.round(e_prod, 1)]

    xs_markers = [[0.0, + marker_width],
                  [1.0, 1.0 + marker_width], [2.0 - marker_width, 2.0]]
    ys_markers = [[ys[0], ys[0]], [ys[1], ys[1]], [ys[2], ys[2]]]

    xs_joins = [[marker_width, 1.0],  [1.0 + marker_width, 2.0 - marker_width]]
    ys_joins = [[ys[0], ys[1]], [ys[1], ys[2]]]

    fig, ax = plt.subplots()
    [ax.plot(xs_markers[i], ys_markers[i], lw=3.0, c='k')
     for i in range(len(xs_markers))]
    [ax.plot(xs_joins[i], ys_joins[i], ls='--', c='k')
     for i in range(len(xs_joins))]

    for i, txt in enumerate(ys):
        ax.annotate(txt, (xs[i], ys[i] + 0.02*max(ys)), fontsize=12)

    if not is_true_ts:
        ax.annotate('TS has >1 imaginary frequency',
                    (1.0, 0.1*max(ys)), ha='center', color='red')
    if not ts_is_converged:
        ax.annotate('TS is not fully converged',
                    (1.0, 0.2*max(ys)), ha='center', color='red')

    plt.title(name, fontdict={'fontsize': 12})
    plt.xticks([])

    if units == KjMol:
        plt.ylabel('∆$E$ / kJ mol$^{-1}$', fontsize=12)
    if units == KcalMol:
        plt.ylabel('∆$E$/ kcal mol$^{-1}$', fontsize=12)

    plt.ylim(min(ys) - 0.05*max(ys), 1.2 * max(ys))
    plt.savefig('reaction_profile.png')

    return 0


def make_reaction_animation(name, xyzs):
    """makes an xyz file that animates the reaction pathway

    Arguments:
        name {str} -- name of the xyz file to be created
        xyzs {list of lists} -- list with each element of the list a list of xyzs
    """
    logger.info('Generating a reaction pathway animation')
    with open(f'{name}_animation.xyz', 'w') as output_file:
        for frame, xyz_list in enumerate(xyzs):
            print(len(xyz_list), file=output_file)
            print(frame, file=output_file)
            [print('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(
                *line), file=output_file) for line in xyz_list]
