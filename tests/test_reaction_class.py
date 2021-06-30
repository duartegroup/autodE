import os
from autode.reactions import reaction
from autode.reactions import reaction_types
from autode.transition_states.transition_state import TransitionState
from autode.bond_rearrangement import BondRearrangement
from autode.species import Reactant, Product
from autode.transition_states.ts_guess import TSguess
from autode.species.complex import ReactantComplex, ProductComplex
from autode.atoms import Atom
from autode.exceptions import UnbalancedReaction
from autode.exceptions import SolventsDontMatch
from autode.mol_graphs import make_graph
from autode.plotting import plot_reaction_profile
from autode.units import KcalMol
from autode.methods import get_hmethod
from autode.config import Config
from .testutils import work_in_zipped_dir
import shutil
import pytest

here = os.path.dirname(os.path.abspath(__file__))

h1 = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)])

h2 = reaction.Reactant(name='h2', atoms=[Atom('H', 1.0, 0.0, 0.0)])
h2_product = reaction.Product(name='h2', atoms=[Atom('H', 1.0, 0.0, 0.0)])

lin_h3 = reaction.Reactant(name='h3_linear', atoms=[Atom('H', -1.76172, 0.79084, -0.00832),
                                                    Atom('H', -2.13052, 0.18085, 0.00494),
                                                    Atom('H', -1.39867, 1.39880, -0.00676)])

trig_h3 = reaction.Product(name='h3_trigonal', atoms=[Atom('H', -1.76172, 0.79084, -0.00832),
                                                      Atom('H', -1.65980, 1.15506, 0.61469),
                                                      Atom('H', -1.39867, 1.39880, -0.00676)])


def test_reaction_class():
    h1 = reaction.Reactant(name='h1', atoms=[Atom('H', 0.0, 0.0, 0.0)])
    hh_product = reaction.Product(name='hh', atoms=[Atom('H', 0.0, 0.0, 0.0),
                                                    Atom('H', 0.7, 0.0, 0.0)])

    # h + h > mol
    hh_reac = reaction.Reaction(h1, h2, hh_product, name='h2_assoc')

    h1.energy = 2
    h2.energy = 3
    hh_product.energy = 1

    # Only swap to dissociation in invoking locate_ts()
    assert hh_reac.type == reaction_types.Addition
    assert len(hh_reac.prods) == 1
    assert len(hh_reac.reacs) == 2
    assert hh_reac.ts is None
    assert hh_reac.tss is None
    assert hh_reac.name == 'h2_assoc'
    assert hh_reac.calc_delta_e() == -4

    h1 = reaction.Reactant(name='h1', atoms=[Atom('H')])
    hh_reactant = reaction.Reactant(name='hh', atoms=[Atom('H'),
                                                      Atom('H', x=1.0)])
    hh_product = reaction.Product(name='hh', atoms=[Atom('H'),
                                                    Atom('H', x=1.0)])

    # h + mol > mol + h
    h_sub = reaction.Reaction(h1, hh_reactant, h2_product, hh_product,
                              solvent_name='water')

    assert h_sub.type == reaction_types.Substitution
    assert h_sub.name == 'reaction'
    assert h_sub.solvent.name == 'water'
    assert h_sub.solvent.smiles == 'O'


def test_check_rearrangement():

    # Linear H3 -> Trigonal H3
    make_graph(species=trig_h3, allow_invalid_valancies=True)
    reac = reaction.Reaction(lin_h3, trig_h3)

    # Should switch reactants and products if the products have more bonds than
    # the reactants, but only when the TS is attempted to be located..

    # assert reac.reacs[0].name == 'h3_trigonal'
    # assert reac.prods[0].name == 'h3_linear'


def test_check_solvent():

    r = Reactant(name='r', solvent_name='water')
    p = Product(name='p')

    with pytest.raises(SolventsDontMatch):
        _ = reaction.Reaction(r, p)

    p = Product(name='p', solvent_name='water')
    reaction_check = reaction.Reaction(r, p)
    assert reaction_check.solvent.name == 'water'


def test_reaction_identical_reac_prods():

    hh_reactant = reaction.Reactant(name='hh', atoms=[Atom('H'),
                                                      Atom('H', x=1.0)])
    hh_product = reaction.Product(name='hh', atoms=[Atom('H'),
                                                    Atom('H', x=1.0)])

    h2_reaction = reaction.Reaction(hh_reactant, hh_product)

    with pytest.raises(ValueError):
        h2_reaction.locate_transition_state()


def test_swap_reacs_prods():

    reactant = Reactant(name='r')
    product = Product(name='p')

    swapped_reaction = reaction.Reaction(reactant, product)
    assert swapped_reaction.reacs[0].name == 'r'
    assert swapped_reaction.prods[0].name == 'p'

    swapped_reaction.switch_reactants_products()
    assert swapped_reaction.reacs[0].name == 'p'
    assert swapped_reaction.prods[0].name == 'r'


def test_bad_balance():

    hh_product = reaction.Product(name='hh',
                                  atoms=[Atom('H'), Atom('H', x=1.0)])

    with pytest.raises(UnbalancedReaction):
        reaction.Reaction(h1, hh_product)

    h_minus = reaction.Reactant(name='h1_minus', atoms=[Atom('H')], charge=-1)
    with pytest.raises(UnbalancedReaction):
        reaction.Reaction(h1, h_minus, hh_product)

    h1_water = reaction.Reactant(name='h1', atoms=[Atom('H')],
                                 solvent_name='water')
    h2_water = reaction.Reactant(name='h2', atoms=[Atom('H', x=1.0)],
                                 solvent_name='water')
    hh_thf = reaction.Product(name='hh', atoms=[Atom('H'), Atom('H', x=1.0)],
                              solvent_name='thf')

    with pytest.raises(SolventsDontMatch):
        reaction.Reaction(h1_water, h2_water, hh_thf)

    with pytest.raises(NotImplementedError):
        hh_triplet = reaction.Product(name='hh_trip',
                                      atoms=[Atom('H'), Atom('H', x=0.7)],
                                      mult=3)
        reaction.Reaction(h1, h2, hh_triplet)


def test_calc_delta_e():

    r1 = reaction.Reactant(name='h', atoms=[Atom('H')])
    r1.energy = -0.5

    r2 = reaction.Reactant(name='h', atoms=[Atom('H')])
    r2.energy = -0.5

    tsguess = TSguess(atoms=None, reactant=ReactantComplex(r1),
                      product=ProductComplex(r2))
    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    ts.energy = -0.8

    p = reaction.Product(name='hh', atoms=[Atom('H'), Atom('H', x=1.0)])
    p.energy = -1.0

    reac = reaction.Reaction(r1, r2, p)
    reac.ts = ts

    assert -1E-6 < reac.calc_delta_e() < 1E-6
    assert 0.2 - 1E-6 < reac.calc_delta_e_ddagger() < 0.2 + 1E-6


def test_from_smiles():
    # Chemdraw can generate a reaction with reactants and products
    addition = reaction.Reaction(smiles='CC(C)=O.[C-]#N>>CC([O-])(C#N)C')

    assert len(addition.reacs) == 2
    assert len(addition.prods) == 1

    # Should be readable-ish names
    for reac in addition.reacs:
        assert reac.name != 'molecule'

    with pytest.raises(UnbalancedReaction):
        _ = reaction.Reaction('CC(C)=O.[C-]#N')


def test_single_points():

    # Spoof ORCA install
    Config.ORCA.path = here

    rxn = reaction.Reaction(Reactant(smiles='O'), Product(smiles='O'))

    # calculate_single_points should be pretty tolerant.. not raising
    # exceptions if the energy is already None
    rxn.calculate_single_points()
    assert rxn.reacs[0].energy is None

    overlapping_h2 = Reactant(atoms=[Atom('H'), Atom('H')])
    overlapping_h2.energy = -1
    rxn.reacs = [overlapping_h2]

    # Shouldn't calculate a single point for a molecule that is not
    # 'reasonable'
    rxn.calculate_single_points()
    assert rxn.reacs[0].energy == -1

    Config.ORCA.path = None


@work_in_zipped_dir(os.path.join(here, 'data', 'free_energy_profile.zip'))
def test_free_energy_profile():

    # Use a spoofed Gaussian09 and XTB install
    Config.lcode = 'xtb'

    Config.hcode = 'g09'
    Config.G09.path = here

    Config.ts_template_folder_path = os.getcwd()
    Config.hmethod_conformers = False
    Config.standard_state = '1atm'
    Config.lfm_method = 'igm'

    method = get_hmethod()
    assert method.name == 'g09'
    assert method.available

    rxn = reaction.Reaction(Reactant(name='F-', smiles='[F-]'),
                            Reactant(name='CH3Cl', smiles='ClC'),
                            Product(name='Cl-', smiles='[Cl-]'),
                            Product(name='CH3F', smiles='CF'),
                            name='sn2', solvent_name='water')

    # Don't run the calculation without a working XTB install
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    rxn.calculate_reaction_profile(free_energy=True)

    # Allow ~0.5 kcal mol-1 either side of the 'true' value

    dg_ts = rxn.calc_delta_g_ddagger()
    assert 16 < dg_ts.to('kcal mol-1') < 18

    dg_r = rxn.calc_delta_g()
    assert -14 < dg_r.to('kcal mol-1') < -12

    dh_ts = rxn.calc_delta_h_ddagger()
    assert 9 < dh_ts.to('kcal mol-1') < 11

    dh_r = rxn.calc_delta_h()
    assert -14 < dh_r.to('kcal mol-1') < -12

    # Should be able to plot an enthalpy profile
    plot_reaction_profile([rxn], units=KcalMol, name='enthalpy',
                          enthalpy=True)
    assert os.path.exists('enthalpy_reaction_profile.png')
    os.remove('enthalpy_reaction_profile.png')

    # Reset the configuration to the default values
    Config.hcode = None
    Config.G09.path = None
    Config.lcode = None
    Config.XTB.path = None


def test_unavail_properties():
    ha = reaction.Reactant(name='ha', atoms=[Atom('H')])

    hb = reaction.Product(name='hb', atoms=[Atom('H')])

    rxn = reaction.Reaction(ha, hb)
    delta = reaction.calc_delta_with_cont(left=[ha], right=[hb], cont='h_cont')
    assert delta is None

    # Should not raise an exception(?)
    rxn.find_lowest_energy_ts_conformer()
    rxn.calculate_thermochemical_cont(free_energy=False, enthalpy=False)
