import os
import numpy as np
from time import time
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
from autode.values import (
    PotentialEnergy,
    FreeEnergy,
    Enthalpy,
    EnthalpyCont,
    FreeEnergyCont,
)
from autode.methods import get_hmethod
from autode.config import Config
from .testutils import work_in_zipped_dir, requires_with_working_xtb_install
import pytest

here = os.path.dirname(os.path.abspath(__file__))

# Spoof ORCA install
Config.hcode = "ORCA"
Config.ORCA.path = here

h1 = Reactant(name="h1", atoms=[Atom("H", 0.0, 0.0, 0.0)])

h2 = Reactant(name="h2", atoms=[Atom("H", 1.0, 0.0, 0.0)])
h2_product = Product(name="h2", atoms=[Atom("H", 1.0, 0.0, 0.0)])

lin_h3 = Reactant(
    name="h3_linear",
    atoms=[
        Atom("H", -1.76172, 0.79084, -0.00832),
        Atom("H", -2.13052, 0.18085, 0.00494),
        Atom("H", -1.39867, 1.39880, -0.00676),
    ],
)

trig_h3 = Product(
    name="h3_trigonal",
    atoms=[
        Atom("H", -1.76172, 0.79084, -0.00832),
        Atom("H", -1.65980, 1.15506, 0.61469),
        Atom("H", -1.39867, 1.39880, -0.00676),
    ],
)


def test_reaction_class():

    h1 = reaction.Reactant(name="h1", atoms=[Atom("H", 0.0, 0.0, 0.0)])
    hh_product = reaction.Product(
        name="hh", atoms=[Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.7, 0.0, 0.0)]
    )

    # h + h > mol
    hh_reac = reaction.Reaction(h1, h2, hh_product, name="h2_assoc")

    h1.energy = 2
    h2.energy = 3
    hh_product.energy = 1

    assert hh_reac.atomic_symbols == ["H", "H"]

    # Only swap to dissociation in invoking locate_ts()
    assert hh_reac.type == reaction_types.Addition
    assert len(hh_reac.prods) == 1
    assert len(hh_reac.reacs) == 2
    assert hh_reac.ts is None
    assert len(hh_reac.tss) == 0
    assert hh_reac.name == "h2_assoc"
    assert hh_reac.delta("E") == PotentialEnergy(-4.0)

    h1 = reaction.Reactant(name="h1", atoms=[Atom("H")])
    hh_reactant = reaction.Reactant(
        name="hh", atoms=[Atom("H"), Atom("H", x=1.0)]
    )
    hh_product = reaction.Product(
        name="hh", atoms=[Atom("H"), Atom("H", x=1.0)]
    )

    # h + mol > mol + h
    h_sub = reaction.Reaction(
        h1, hh_reactant, h2_product, hh_product, solvent_name="water"
    )

    assert h_sub.type == reaction_types.Substitution
    assert h_sub.name == "reaction"
    assert h_sub.solvent.name == "water"
    assert h_sub.solvent.smiles == "O"
    for mol in h_sub.reacs + h_sub.prods:
        assert mol.solvent.name == "water"

    # Must set the transition state with a TransitionState
    with pytest.raises(ValueError):
        h_sub.ts = h1


def test_reactant_product_complexes():

    h2_prod = Product(name="h2", atoms=[Atom("H"), Atom("H", x=1.0)])

    rxn = reaction.Reaction(h1, h2, h2_prod)
    assert rxn.reactant.n_molecules == 2
    assert rxn.reactant.distance(0, 1) > 1

    assert rxn.product.n_molecules == 1

    # If the reactant complex is set then the whole reactant should be that
    rxn.reactant = ReactantComplex(
        h1, h1, copy=True, do_init_translation=False
    )
    assert -1e-4 < rxn.reactant.distance(0, 1) < 1e-4

    # but cannot be just a reactant
    with pytest.raises(ValueError):
        rxn.reactant = h1

    # and similarly with the products
    with pytest.raises(ValueError):
        rxn.product = h2

    # but can set the product complex
    rxn.product = ProductComplex(
        Product(atoms=[Atom("H"), Atom("H", x=1.0)]), name="tmp"
    )
    assert rxn.product.name == "tmp"


def test_invalid_with_complexes():

    Config.hcode = "ORCA"
    Config.ORCA.path = here

    h3_reaction = reaction.Reaction(lin_h3, trig_h3)

    # Currently free energies with association complexes is not supported
    with pytest.raises(NotImplementedError):
        h3_reaction.calculate_reaction_profile(
            with_complexes=True, free_energy=True
        )

    # Cannot plot a reaction profile with complexes without them existing
    with pytest.raises(ValueError):
        h3_reaction._plot_reaction_profile_with_complexes(
            units=KcalMol, free_energy=False, enthalpy=False
        )


def test_check_rearrangement():

    # Linear H3 -> Trigonal H3
    make_graph(species=trig_h3, allow_invalid_valancies=True)
    reac = reaction.Reaction(lin_h3, trig_h3)

    # Should switch reactants and products if the products have more bonds than
    # the reactants, but only when the TS is attempted to be located..

    # assert reac.reacs[0].name == 'h3_trigonal'
    # assert reac.prods[0].name == 'h3_linear'


def test_check_solvent():

    r = Reactant(name="r", solvent_name="water")
    p = Product(name="p")

    with pytest.raises(SolventsDontMatch):
        _ = reaction.Reaction(r, p)

    p = Product(name="p", solvent_name="water")
    reaction_check = reaction.Reaction(r, p)
    assert reaction_check.solvent.name == "water"


def test_reaction_identical_reac_prods():

    Config.hcode = "ORCA"
    Config.ORCA.path = here

    hh_reactant = reaction.Reactant(
        name="hh", atoms=[Atom("H"), Atom("H", x=1.0)]
    )
    hh_product = reaction.Product(
        name="hh", atoms=[Atom("H"), Atom("H", x=1.0)]
    )

    h2_reaction = reaction.Reaction(hh_reactant, hh_product)

    with pytest.raises(ValueError):
        h2_reaction.locate_transition_state()


def test_swap_reacs_prods():

    reactant = Reactant(name="r")
    product = Product(name="p")

    swapped_reaction = reaction.Reaction(reactant, product)
    assert swapped_reaction.reacs[0].name == "r"
    assert swapped_reaction.prods[0].name == "p"

    swapped_reaction.switch_reactants_products()
    assert swapped_reaction.reacs[0].name == "p"
    assert swapped_reaction.prods[0].name == "r"


def test_bad_balance():

    hh_product = reaction.Product(
        name="hh", atoms=[Atom("H"), Atom("H", x=1.0)]
    )

    with pytest.raises(UnbalancedReaction):
        reaction.Reaction(h1, hh_product)

    h_minus = reaction.Reactant(name="h1_minus", atoms=[Atom("H")], charge=-1)
    with pytest.raises(UnbalancedReaction):
        reaction.Reaction(h1, h_minus, hh_product)

    h1_water = reaction.Reactant(
        name="h1", atoms=[Atom("H")], solvent_name="water"
    )
    h2_water = reaction.Reactant(
        name="h2", atoms=[Atom("H", x=1.0)], solvent_name="water"
    )
    hh_thf = reaction.Product(
        name="hh", atoms=[Atom("H"), Atom("H", x=1.0)], solvent_name="thf"
    )

    with pytest.raises(SolventsDontMatch):
        reaction.Reaction(h1_water, h2_water, hh_thf)

    with pytest.raises(NotImplementedError):
        hh_triplet = reaction.Product(
            name="hh_trip", atoms=[Atom("H"), Atom("H", x=0.7)], mult=3
        )
        reaction.Reaction(h1, h2, hh_triplet)


def test_calc_delta_e():

    r1 = reaction.Reactant(name="h", atoms=[Atom("H")])
    r1.energy = -0.5

    r2 = reaction.Reactant(name="h", atoms=[Atom("H")])
    r2.energy = -0.5

    reac_complex = ReactantComplex(r1)
    assert reac_complex.graph is not None

    tsguess = TSguess(
        atoms=reac_complex.atoms,
        reactant=reac_complex,
        product=ProductComplex(r2.to_product()),
    )

    tsguess.bond_rearrangement = BondRearrangement()
    ts = TransitionState(tsguess)
    ts.energy = -0.8

    p = reaction.Product(name="hh", atoms=[Atom("H"), Atom("H", x=1.0)])
    p.energy = -1.0

    reac = reaction.Reaction(r1, r2, p)
    reac.ts = ts

    assert -1e-6 < reac.delta("E") < 1e-6
    assert 0.2 - 1e-6 < reac.delta("E‡") < 0.2 + 1e-6


def test_from_smiles():
    # Chemdraw can generate a reaction with reactants and products
    addition = reaction.Reaction(smiles="CC(C)=O.[C-]#N>>CC([O-])(C#N)C")

    assert len(addition.reacs) == 2
    assert len(addition.prods) == 1

    # Should be readable-ish names
    for reac in addition.reacs:
        assert reac.name != "molecule"

    with pytest.raises(UnbalancedReaction):
        _ = reaction.Reaction("CC(C)=O.[C-]#N")


def test_single_points():

    # Spoof ORCA install
    Config.ORCA.path = here

    rxn = reaction.Reaction(Reactant(smiles="O"), Product(smiles="O"))

    # calculate_single_points should be pretty tolerant.. not raising
    # exceptions if the energy is already None
    rxn.calculate_single_points()
    assert rxn.reacs[0].energy is None

    overlapping_h2 = Reactant(atoms=[Atom("H"), Atom("H")])
    overlapping_h2.energy = -1
    rxn.reacs = [overlapping_h2]

    # Shouldn't calculate a single point for a molecule that is not
    # 'reasonable'
    rxn.calculate_single_points()
    assert rxn.reacs[0].energy == -1

    Config.ORCA.path = None


@work_in_zipped_dir(os.path.join(here, "data", "free_energy_profile.zip"))
@requires_with_working_xtb_install
def test_free_energy_profile():

    # Use a spoofed Gaussian09 and XTB install
    Config.lcode = "xtb"

    Config.hcode = "g09"
    Config.G09.path = here

    Config.ts_template_folder_path = os.getcwd()
    Config.hmethod_conformers = False
    Config.standard_state = "1atm"
    Config.lfm_method = "igm"

    method = get_hmethod()
    assert method.name == "g09"
    assert method.is_available

    rxn = reaction.Reaction(
        Reactant(name="F-", smiles="[F-]"),
        Reactant(name="CH3Cl", smiles="ClC"),
        Product(name="Cl-", smiles="[Cl-]"),
        Product(name="CH3F", smiles="CF"),
        name="sn2",
        solvent_name="water",
    )

    start_time = time()
    rxn.calculate_reaction_profile(free_energy=True)
    full_calc_reaction_profile_time = time() - start_time
    rxn.save("tmp.chk")

    # Allow ~0.5 kcal mol-1 either side of the 'true' value

    assert 16 < rxn.delta("G‡").to("kcal mol-1") < 18
    assert -14 < rxn.delta("G").to("kcal mol-1") < -12

    assert 9 < rxn.delta("H‡").to("kcal mol-1") < 11
    assert -14 < rxn.delta("H").to("kcal mol-1") < -12

    # Should be able to plot an enthalpy profile
    plot_reaction_profile([rxn], units=KcalMol, name="enthalpy", enthalpy=True)
    assert os.path.exists("enthalpy_reaction_profile.pdf")
    os.remove("enthalpy_reaction_profile.pdf")

    # Rerunning the reaction should be fast
    start_time = time()
    rxn.calculate_reaction_profile(free_energy=True)
    assert time() - start_time < full_calc_reaction_profile_time / 2

    # Should be able to reload the entire reaction state
    reloaded_rxn = reaction.Reaction.from_checkpoint("tmp.chk")
    assert reloaded_rxn.ts is not None

    # Reset the configuration to the default values
    Config.hcode = None
    Config.G09.path = None
    Config.lcode = None
    Config.XTB.path = None


def test_barrierless_rearrangment():

    rxn = reaction.Reaction(Reactant(), Product())
    assert rxn.is_barrierless
    assert rxn.delta("E") is rxn.delta("E‡") is None

    a = Reactant(atoms=[Atom("H"), Atom("H", x=-1.0), Atom("H", x=1.0)])
    a.energy = -2.0

    b = Product(atoms=[Atom("H"), Atom("H", x=0.7, y=0.7), Atom("H", x=1.0)])
    b.energy = -2.5

    rxn = reaction.Reaction(a, b)
    # Barrier should be ~0 for an exothermic reaction
    assert rxn.delta("E‡") == 0.0

    # but the reaction energy for an endothermic reaction
    b.energy = -1.5
    assert rxn.delta("E‡") == 0.5


def test_doc_example():
    """If this test changes PLEASE update the documentation at the same time"""

    ethene = Reactant(smiles="C=C")
    butadiene = Reactant(smiles="C=CC=C")
    cyclohexene = Product(smiles="C1=CCCCC1")

    rxn = reaction.Reaction(ethene, butadiene, cyclohexene)
    assert rxn.solvent is None

    assert np.isclose(rxn.temp, 298.15)

    assert rxn.delta("E") is None
    assert rxn.delta("E‡") is None

    # Only allow some indication of energy/enthalpy/free energy
    with pytest.raises(ValueError):
        _ = rxn.delta("X")

    ethene.energy = -6.27126052543
    butadiene.energy = -11.552702027244
    cyclohexene.energy = -17.93143795711

    assert np.isclose(
        float(rxn.delta("E").to("kcal mol-1")), -67.441, atol=0.01
    )

    # Should allow for aliases of the kind of ∆ difference
    assert rxn.delta("E") == rxn.delta("energy")
    assert (
        rxn.delta("G") == rxn.delta("free energy") == rxn.delta("free_energy")
    )
    assert rxn.delta("H") == rxn.delta("enthalpy") != rxn.delta("energy")

    assert np.isclose(float(rxn.delta("E‡").to("kcal mol-1")), 4.35491, atol=1)

    # Post-optimisation
    cyclohexene.energy = -234.206929484613
    butadiene.energy = -155.686225567141
    ethene.energy = -78.427547239225

    atoms = [
        Atom("C", 1.54832502175757, 0.47507857149246, -0.17608869645477),
        Atom("C", 0.63680758724295, 1.48873574610658, 0.18687763919775),
        Atom("C", -0.48045277463960, 1.22774144243181, 0.94515144035260),
        Atom("C", -1.56617230970127, -0.32194220965435, -0.36403038119916),
        Atom("C", -0.67418142609443, -1.30824343582455, -0.72092960944319),
        Atom("C", 1.37354885332651, -0.83414272445258, 0.20733555387781),
        Atom("H", 2.28799156107427, 0.70407896562267, -0.95011846456488),
        Atom("H", 0.71504911441008, 2.45432905429025, -0.32320231835197),
        Atom("H", -1.22975529929055, 2.00649252772661, 1.11086944491345),
        Atom("H", -0.48335986214289, 0.41671089728999, 1.67449683583301),
        Atom("H", -2.28180803085362, -0.49278222876383, 0.44394762543300),
        Atom("H", -1.83362998587976, 0.46266072581502, -1.07348798508224),
        Atom("H", -0.67829969484189, -2.26880762586176, -0.19982465470916),
        Atom("H", -0.22671274865021, -1.31325250991020, -1.71602826881263),
        Atom("H", 0.86381884115892, -1.08003298402692, 1.13995512315906),
        Atom("H", 2.02879115312393, -1.61656419228120, -0.18499328414869),
    ]

    rxn.ts = TransitionState(TSguess(atoms=atoms))
    rxn.ts.energy = -234.090983203239

    assert "TransitionState" in repr(rxn.ts)

    assert np.isclose(float(rxn.delta("E‡").to("kcal mol-1")), 14.3, atol=0.1)


def test_barrierless_h_g():

    a = Reactant(atoms=[Atom("H"), Atom("H", x=-1.0), Atom("H", x=1.0)])
    a.energies.extend(
        [PotentialEnergy(-1), EnthalpyCont(0.1), FreeEnergyCont(0.3)]
    )

    b = Product(atoms=[Atom("H"), Atom("H", x=0.7, y=0.7), Atom("H", x=1.0)])
    b.energies.extend(
        [PotentialEnergy(-2), EnthalpyCont(0.2), FreeEnergyCont(0.6)]
    )

    rxn = reaction.Reaction(a, b)
    assert rxn.delta("E‡") == 0.0
    assert rxn.delta("H‡") == 0.0
    assert rxn.delta("G‡") == 0.0

    rxn.switch_reactants_products()
    assert np.isclose(rxn.delta("E‡"), 1.0)

    assert np.isclose(rxn.delta("H‡"), 0.9)  # -2+0.2 -> -1+0.1   --> ∆ = 0.9

    assert np.isclose(rxn.delta("G‡"), 0.7)  # -2+0.6 -> -1+0.3   --> ∆ = 0.7


def test_same_composition():

    r1 = reaction.Reaction(
        Reactant(atoms=[Atom("C"), Atom("H", x=1)]),
        Product(atoms=[Atom("C"), Atom("H", x=10)]),
    )

    r2 = reaction.Reaction(
        Reactant(atoms=[Atom("C"), Atom("H", x=1)]),
        Product(atoms=[Atom("C"), Atom("H", x=5)]),
    )

    assert r1.has_identical_composition_as(r2)

    r3 = reaction.Reaction(
        Reactant(name="h2", atoms=[Atom("H", 1.0, 0.0, 0.0)]),
        Product(name="h2", atoms=[Atom("H", 1.0, 0.0, 0.0)]),
    )
    assert not r1.has_identical_composition_as(r3)


def test_name_uniqueness():

    rxn = reaction.Reaction(
        Reactant(smiles="CC[C]([H])[H]"), Product(smiles="C[C]([H])C")
    )

    assert rxn.reacs[0].name != rxn.prods[0].name


def test_identity_reaction_is_supported_with_labels():
    def reaction_is_isomorphic(_r):
        return _r.reactant.graph.is_isomorphic_to(_r.product.graph)

    isomorphic_rxn = reaction.Reaction("[Br-].C[Br]>>C[Br].[Br-]")
    assert reaction_is_isomorphic(isomorphic_rxn)

    rxn = reaction.Reaction("[Br-:1].C[Br:2]>>C[Br:1].[Br-:2]")
    assert not reaction_is_isomorphic(rxn)


def test_cannot_run_locate_ts_with_no_reactants_or_products():

    Config.lcode = Config.hcode = "ORCA"
    Config.ORCA.path = here

    rxn = reaction.Reaction()
    with pytest.raises(RuntimeError):
        rxn.locate_transition_state()

    Config.lcode = None
