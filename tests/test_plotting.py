from autode import plotting
import matplotlib.pyplot as plt
from autode.input_output import xyz_file_to_atoms
from autode.exceptions import CouldNotPlotSmoothProfile
from autode.species.molecule import Reactant, Product
from autode.calculations import Calculation
from autode.methods import ORCA
from autode.transition_states.transition_state import TransitionState
from autode.species.complex import ReactantComplex, ProductComplex
from autode.reactions.reaction import Reaction
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.ts_guess import TSguess
from autode.units import KjMol, KcalMol
from autode.config import Config
from copy import deepcopy
from scipy.optimize import minimize
from scipy import interpolate
from . import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))
Config.high_quality_plots = False


def test_plot_reaction_profile():

    r = Reactant(name="reactant", smiles="C")
    p = Product(name="product", smiles="C")
    tsguess = TSguess(
        atoms=r.atoms, reactant=ReactantComplex(r), product=ProductComplex(p)
    )
    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    reaction = Reaction(r, p)
    reaction.ts = ts

    plotting.plot_reaction_profile(
        reactions=[reaction], units=KjMol, name="test"
    )

    assert os.path.exists("test_reaction_profile.pdf")
    os.remove("test_reaction_profile.pdf")

    with pytest.raises(AssertionError):
        plotting.plot_reaction_profile(
            reactions=[reaction],
            units=KjMol,
            name="test",
            free_energy=True,
            enthalpy=True,
        )
    return None


def test_stat_points():

    # y = (x-2)^2  has a stationary point at x = 2

    stationary_points = plotting.get_stationary_points(
        xs=np.linspace(-1, 3, 100), dydx=lambda x: 2 * (x - 2)
    )

    assert len(stationary_points) == 1
    assert 1.9 < stationary_points[0] < 2.1


def test_error_on_stat_points():

    energies = np.array([0, 10, 0])

    # Symmetric energy array shpuld give very low difference between the
    # required energies and those obtained at the splined stationary points
    assert plotting.error_on_stationary_points(energies, energies) < 1e-3


def test_calculate_reaction_profile_energies():

    test_reac = Reactant(name="test", smiles="C")
    test_reac.energy = -1

    test_prod = Product(name="test", smiles="C")
    test_prod.energy = -1.03187251

    tsguess = TSguess(
        atoms=test_reac.atoms,
        reactant=ReactantComplex(test_reac),
        product=ProductComplex(),
    )

    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    ts.energy = -0.96812749

    reaction = Reaction(test_reac, test_prod)
    reaction.ts = ts

    energies = plotting.calculate_reaction_profile_energies(
        reactions=[reaction], units=KcalMol
    )

    # Energies have been set to ∆E = -20 and ∆E‡ = 20 kcal mol-1 respectively
    assert energies[0] == 0
    assert 19 < energies[1] < 21
    assert -21 < energies[2] < -19

    # Copying the reaction should give relative energies [0, 20, -20, 0, -40]

    energies = plotting.calculate_reaction_profile_energies(
        reactions=[reaction, deepcopy(reaction)], units=KcalMol
    )

    # Energies have been set to ∆E = -20 and ∆E‡ = 20 kcal mol-1 respectively
    assert energies[0] == 0
    assert -0.1 < energies[3] < 0.1
    assert -41 < energies[4] < -39


@testutils.work_in_zipped_dir(os.path.join(here, "data", "plotting.zip"))
def test_reaction_warnings():
    test_reac = Reactant(name="test", smiles="C")
    test_reac.energy = -1

    test_prod = Product(name="test", smiles="C")
    test_prod.energy = -1.03187251

    tsguess = TSguess(
        atoms=test_reac.atoms,
        reactant=ReactantComplex(test_reac),
        product=ProductComplex(),
    )
    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    ts.energy = -0.98

    reaction = Reaction(test_reac, test_prod)
    reaction.ts = None

    # Should be some warning with no TS
    warning_str = plotting.get_reaction_profile_warnings(reactions=[reaction])
    assert len(warning_str) > 10
    assert "barrierless" in warning_str

    # Should be no warnings  with a TS that exists and has an energy and one
    # imaginary freq
    ts.atoms = xyz_file_to_atoms("TS.xyz")
    ts.charge = -1
    orca = ORCA()
    ts_calc = Calculation(
        name="TS", molecule=ts, method=orca, keywords=orca.keywords.opt_ts
    )
    ts_calc.set_output_filename("TS.out")
    reaction.ts = ts
    warnings = plotting.get_reaction_profile_warnings(reactions=[reaction])
    assert "None" in warnings


def test_edge_case_plot():

    # Some inputs cannot be plotted as a smooth profile as optimisation of the
    # energies to get the correct stationary values removes some stationary
    # points

    with pytest.raises(CouldNotPlotSmoothProfile):
        energies = np.array([0.0, 4.0, 0.05, -16, 0.3])
        fig, ax = plt.subplots()

        plotting.plot_smooth_profile(
            zi_s=np.array([0, 1, 2, 3, 4]), energies=energies, ax=ax
        )

    # But should be able to just plot the points connected by lines
    fig, ax = plt.subplots()
    plotting.plot_points(
        zi_s=np.array([0, 1, 2, 3, 4]), energies=energies, ax=ax
    )
    plt.close()


def test_stat_point_minimisation():
    # Test that the minimisation works for very shallow minima

    energies_list = [
        np.array([0.0, 3.8, -9.1, -1.6, 0.3]),
        np.array([0.0, 10, -20, 10, -5]),
    ]

    for energies in energies_list:

        result = minimize(
            plotting.error_on_stationary_points,
            x0=energies,
            args=(energies,),
            method="BFGS",
            tol=0.1,
        )

        assert result.success

        spline = interpolate.CubicSpline(
            [0, 1, 2, 3, 4], result.x, bc_type="clamped"
        )
        fine_zi_s = np.linspace(-0.2, 5.2, num=500)
        stationary_points = plotting.get_stationary_points(
            xs=fine_zi_s, dydx=spline.derivative()
        )
        assert len(stationary_points) == 5


def test_energy():

    energy = plotting.Energy(5, units="Ha", estimated=False)
    assert not energy.is_estimated
    assert energy == 5

    new_energy = energy * 5
    assert new_energy == 25
    assert not new_energy.is_estimated

    new_energy = energy - 5
    assert new_energy == 0

    energy2 = plotting.Energy(2, units="Ha", estimated=False)
    new_energy = 5 * energy2
    assert new_energy == 10

    assert "energy" in repr(energy).lower()
