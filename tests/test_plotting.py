from autode import plotting
from autode.molecule import Reactant, Product
from autode.transition_states.transition_state import TransitionState
from autode.complex import ReactantComplex, ProductComplex
from autode.reaction import Reaction
from autode.bond_rearrangement import BondRearrangement
from autode.transition_states.ts_guess import TSguess
from autode.units import KjMol, KcalMol
from autode.config import Config
from copy import deepcopy
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
Config.high_quality_plots = False


def test_plot_reaction_profile():
    # only tests the file is created with the right name
    os.chdir(os.path.join(here, 'data'))

    r = Reactant(name='reactant', smiles='C')
    p = Product(name='product', smiles='C')
    tsguess = TSguess(atoms=r.atoms, reactant=ReactantComplex(r), product=ProductComplex(p))
    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    reaction = Reaction(r, p)
    reaction.ts = ts

    plotting.plot_reaction_profile(reactions=[reaction], units=KjMol, name='test_reaction')

    assert os.path.exists('test_reaction_reaction_profile.png')

    os.remove('test_reaction_reaction_profile.png')
    os.chdir(here)


def test_stat_points():

    # y = (x-2)^2  has a stationary point at x = 2

    stationary_points = plotting.get_stationary_points(xs=np.linspace(-1, 3, 100),
                                                       dydx=lambda x: 2*(x-2))

    assert len(stationary_points) == 1
    assert 1.9 < stationary_points[0] < 2.1


def test_error_on_stat_points():

    energies = np.array([0, 10, 0])

    # Symmetric energy array shpuld give very low difference between the required energies
    # and those obtained at the splined stationary points
    assert plotting.error_on_stationary_points(energies, energies) < 1E-3


def test_calulcate_reaction_profile_energies():

    test_reac = Reactant(name='test', smiles='C')
    test_reac.energy = -1

    test_prod = Product(name='test', smiles='C')
    test_prod.energy = -1.03187251

    tsguess = TSguess(atoms=test_reac.atoms, reactant=ReactantComplex(test_reac), product=ProductComplex())
    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    ts.energy = -0.96812749

    reaction = Reaction(test_reac, test_prod)
    reaction.ts = ts

    energies = plotting.calculate_reaction_profile_energies(reactions=[reaction],
                                                            units=KcalMol)

    # Energies have been set to ∆E = -20 and ∆E‡ = 20 kcal mol-1 respectively
    assert energies[0] == 0
    assert 19 < energies[1] < 21
    assert -21 < energies[2] < -19

    # Copying the reaction should give relative energies [0, 20, -20, 0, -40]

    energies = plotting.calculate_reaction_profile_energies(reactions=[reaction, deepcopy(reaction)],
                                                            units=KcalMol)

    # Energies have been set to ∆E = -20 and ∆E‡ = 20 kcal mol-1 respectively
    assert energies[0] == 0
    assert -0.1 < energies[3] < 0.1
    assert -41 < energies[4] < -39

    # If reactants and products have been switched for a reaction then
    # the energies in the reaction profile should be the correct order
    reaction.switched_reacs_prods = True

    energies = plotting.calculate_reaction_profile_energies(reactions=[reaction],
                                                            units=KcalMol)
    assert energies[0] == 0
    assert 39 < energies[1] < 41
    assert 19 < energies[2] < 21



def test_reaction_warnings():
    test_reac = Reactant(name='test', smiles='C')
    test_reac.energy = -1

    test_prod = Product(name='test', smiles='C')
    test_prod.energy = -1.03187251

    tsguess = TSguess(atoms=test_reac.atoms, reactant=ReactantComplex(test_reac), product=ProductComplex())
    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    ts.energy = -0.98
    ts.imaginary_frequencies = [-100]

    reaction = Reaction(test_reac, test_prod)
    reaction.ts = None

    # Should be some warning with no TS
    assert len(plotting.get_reaction_profile_warnings(reactions=[reaction])) > 10

    # Should be no warnings  with a TS that exists and has an energy and one imaginary freq
    reaction.ts = ts
    warnings = plotting.get_reaction_profile_warnings(reactions=[reaction])
    assert 'None' in warnings
