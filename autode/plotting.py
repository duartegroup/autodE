import os
import numpy as np
from typing import Sequence, Union
from scipy import interpolate
from autode.values import Energy
from autode.exceptions import CouldNotPlotSmoothProfile
from scipy.optimize import minimize
from autode.config import Config
from autode.units import energy_unit_from_name
from autode.log import logger


def save_plot(plot, filename):
    """Save a plot"""

    if os.path.exists(filename):
        logger.warning("Plot already exists. Overriding..")
        os.remove(filename)

    plot.savefig(filename, dpi=400 if Config.high_quality_plots else 100)
    plot.close()

    return None


def plot_reaction_profile(
    reactions: Sequence["autode.reactions.Reaction"],
    units: Union["autode.units.Unit", str],
    name: str,
    free_energy: bool = False,
    enthalpy: bool = False,
) -> None:
    """
    For a set of reactions plot the reaction profile using matplotlib

    ---------------------------------------------------------------------------
    Arguments:
        reactions (list((autode.reaction.Reaction)):

        units (autode.units.Units | str):

        name (str):

        free_energy (bool): Plot the free energy profile (G)

        enthalpy (bool): Plot the enthalpic profile (H)
    """
    import matplotlib.pyplot as plt

    logger.info("Plotting reaction profile")

    if free_energy and enthalpy:
        raise AssertionError("Cannot plot a profile in both G and H")

    if isinstance(units, str):
        units = energy_unit_from_name(name=units)

    fig, ax = plt.subplots()

    # Get the energies for the reaction profile (y values) plotted against the
    # reaction coordinate (zi_s)
    energies = calculate_reaction_profile_energies(
        reactions, units=units, free_energy=free_energy, enthalpy=enthalpy
    )
    zi_s = np.array(range(len(energies)))

    try:
        plot_smooth_profile(zi_s, energies, ax=ax)

    except CouldNotPlotSmoothProfile:
        plot_points(zi_s, energies, ax=ax)

    ec = "E"
    if free_energy:
        ec = "G"
    elif enthalpy:
        ec = "H"

    plt.ylabel(f"∆${ec}$ / {units.plot_name}", fontsize=12)
    plt.xlabel("Reaction coordinate")

    energy_values = [energy for energy in energies]
    max_delta = max(energy_values) - min(energy_values)
    plt.ylim(
        min(energy_values) - 0.09 * max_delta,
        max(energy_values) + 0.09 * max_delta,
    )
    plt.xticks([])
    plt.subplots_adjust(top=0.95, right=0.95)
    fig.text(
        0.1,
        0.05,
        get_reaction_profile_warnings(reactions),
        ha="left",
        fontsize=6,
        wrap=True,
    )

    prefix = "" if name == "reaction" else f"{name}_"
    return save_plot(plt, filename=f"{prefix}reaction_profile.pdf")


def plot_smooth_profile(zi_s, energies, ax):
    """
    Plot a smooth reaction profile by spline interpolation and finding the
    stationary points. This will not afford the correct number of stationary
    points for some energy arrays, so raise an exception if it fails

    ---------------------------------------------------------------------------
    Arguments:
        zi_s (np.ndarray): Estimate of reaction coordinate points

        energies (list(autode.plotting.Energy)): len(energies) = len(zi_s)

        ax (matplotlib.axes.Axes):
    """

    # Minimise a set of spline points so the stationary points have y values
    # given in the energies array
    energies_arr = np.array([energy for energy in energies], dtype="f")
    result = minimize(
        error_on_stationary_points,
        x0=energies_arr,
        args=(energies_arr,),
        method="BFGS",
        tol=0.1,
    )

    # Use the optimised values to construct a spline function that will be
    # plotted
    optimised_spline = interpolate.CubicSpline(
        zi_s, result.x, bc_type="clamped"
    )

    # Create more zi values from slightly before the minimum to slightly after
    # the maximum
    fine_zi_s = np.linspace(min(zi_s) - 0.2, max(zi_s) + 0.2, num=500)

    # The new zi values are the stationary points of the optimised function
    zi_s = get_stationary_points(fine_zi_s, optimised_spline.derivative())

    if len(zi_s) != len(energies):
        raise CouldNotPlotSmoothProfile

    # Plot the function
    ax.plot(fine_zi_s, optimised_spline(fine_zi_s), c="k")
    ax.scatter(zi_s, optimised_spline(zi_s), c="b", zorder=10)

    # Annotate the plot with the relative energies
    max_delta = max(energies) - min(energies)

    for i, energy in enumerate(optimised_spline(zi_s)):
        if energies[i].is_estimated:
            # Don't add estimated energies
            continue

        # Shift the minima labels (even points) below the point and the
        # transition state labels above the point
        shift = -0.07 * max_delta if i % 2 == 0 else 0.03 * max_delta

        ax.annotate(
            f"{energy:.1f}",
            (zi_s[i], energy + shift),
            fontsize=12,
            ha="center",
        )

    return None


def plot_points(zi_s, energies, ax):
    """
    Plot a reaction profile just adding the points to the graph

    ---------------------------------------------------------------------------
    Arguments:
        zi_s (np.ndarray): Estimate of reaction coordinate points

        energies (list(autode.plotting.Energy)): len(energies) = len(zi_s)

        ax (matplotlib.axes.Axes):
    """
    energies_arr = np.array([energy for energy in energies])

    ax.plot(zi_s, energies_arr, ls="--", c="k", marker="o")

    # Annotate the plot with the relative energies
    for i, energy in enumerate(energies):
        if hasattr(energy, "estimated") and energy.is_estimated:
            # Don't add estimated energies
            continue

        ax.annotate(
            f"{np.round(energies_arr[i], 1)}",
            (zi_s[i], energies_arr[i] + 0.7),
            fontsize=12,
            ha="center",
        )
    return None


def get_reaction_profile_warnings(reactions):
    """
    Get a string of warnings for a reaction

    ---------------------------------------------------------------------------
    Arguments:
        reactions (list(autode.reaction.Reaction)):

    Returns:
        (str): List of warnings to annotate the plot with
    """
    logger.info("Getting warnings for reaction profile")
    warnings = ""

    for reaction in reactions:

        if reaction.delta("E") is None:
            warnings += (
                f"∆Er not calculated for {reaction.name}, "
                f"∆Er = 0 assumed. "
            )

        de_ts = reaction.delta("E‡")
        if de_ts is None or (de_ts is not None and de_ts.is_estimated):
            warnings += (
                f"∆E‡ not calculated for {reaction.name}, "
                f"barrierless reaction assumed. "
            )

        if reaction.ts is not None:

            if reaction.ts.has_imaginary_frequencies:
                n_imag_freqs = len(reaction.ts.imaginary_frequencies)

                if n_imag_freqs != 1:
                    warnings += (
                        f"TS for {reaction.name} has {n_imag_freqs} "
                        f"imaginary frequencies. "
                    )

            warnings += reaction.ts.warnings

    # If no strings were added then there are no warnings
    if len(warnings) == 0:
        warnings = "None"

    return f"WARNINGS: {warnings}"


def calculate_reaction_profile_energies(
    reactions, units, free_energy=False, enthalpy=False
):
    """
    Calculate a list of energies comprising the reaction profile

    ---------------------------------------------------------------------------
    Arguments:
        reactions (list(autode.reaction.Reaction)):

        units (autode.units.Units):

    Keyword Arguments:
        free_energy (bool): Calculate ∆Gs

        enthalpy (bool): Calculate ∆Hs

    Returns:
        (np.ndarray(autode.plotting.Energy))
    """
    # Populate a list of reaction relative energies
    # [reactants -> TS -> products], all floats
    reaction_energies = []
    energy_type = "H" if enthalpy else ("G" if free_energy else "E")

    for reaction in reactions:

        de = reaction.delta(energy_type)

        # If ∆Er cannot be calculated then assume isoenergetic and add a
        # warning to the plot
        if de is None:
            de = Energy(0.0, estimated=True)

        de_ts = reaction.delta(f"{energy_type}‡")

        # If there is no ∆E then de_ts could be None. Use the Effective free
        # energy barrier of 4.35 kcal mol-1
        if de_ts is None:
            de_ts = Energy(0.00694, units="Ha", estimated=True)

        reaction_energies.append([Energy(0.0), de_ts, de])

    # Construct the full list of energies, referenced to the first set of
    # reactants
    energies = reaction_energies[0]

    for i in range(1, len(reaction_energies)):
        # Add the energies from the next TS and the next product reaction_
        # energies[i][0] == energies[-1
        energies += [
            reaction_energies[i][1] + energies[-1],
            reaction_energies[i][2] + energies[-1],
        ]

    return [energy * units.conversion for energy in energies]


def get_stationary_points(xs, dydx):
    """
    Compute the productive of the derivative at points x(i-1) and x(i) which
    is negative if there is a point x(k)
    between x(i-1) and x(i) that has dy/dx|x(k) = 0

    ---------------------------------------------------------------------------
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
    Calculate the difference between the stationary points of an interpolated
    function and those observed (given in the energies array). Example::

          |      .
        E |.   / |        The points indicate the true stationary points
          | |_/  |.
          |_____________
                zi

    ---------------------------------------------------------------------------
    Arguments:
        x (np.ndarray): Points that will be splined that generate stationary
                        points that ≈ energies

        energies (np.ndarray): Observed stationary points

    Returns:
        (float): A measure of the error
    """
    # Generate a list of reaction coordinate points - arbitrary units so
    # integers are fine
    zi_s = np.array(range(len(x)))

    # Spline the energies to get a function that has stationary points
    spline = interpolate.CubicSpline(zi_s, x, bc_type="clamped")

    # Calculate the energy values at the stationary points of the function with
    # a fine-ish spacing that extrapolates
    # slightly
    fine_zi_s = np.linspace(min(zi_s) - 0.2, max(zi_s) + 0.2, num=500)
    stationary_points = get_stationary_points(
        xs=fine_zi_s, dydx=spline.derivative()
    )

    if len(stationary_points) != len(energies):
        # TODO make this smooth somehow
        # Energy penalty for not having the required number of
        return 10 * np.abs(len(energies) - len(stationary_points))

    energies_at_stationary_points = [spline(zi) for zi in stationary_points]

    # Return the error as the sum squared difference between the required and
    # the observed stationary point energies
    energy_difference = energies - np.array(energies_at_stationary_points)

    return np.sum(np.square(energy_difference))
