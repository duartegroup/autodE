import pytest
from autode.config import Config
from autode.reactions.reaction import Reaction
from autode.reactions.multistep import MultiStepReaction
from autode.species import Reactant, Product
from autode.atoms import Atom
from autode.wrappers.keywords import cpcm
from . import testutils
import shutil
import os

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, "data", "multistep.zip"))
@testutils.requires_with_working_xtb_install
def test_multistep_reaction():

    Config.num_conformers = 1

    # Spoof installs
    Config.lcode = "xtb"
    Config.XTB.path = shutil.which("xtb")

    Config.hcode = "orca"
    Config.ORCA.path = here

    Config.ORCA.implicit_solvation_type = cpcm
    Config.make_ts_template = False
    Config.num_complex_sphere_points = 2
    Config.num_complex_random_rotations = 1

    # SN2 forwards then backwards example
    forwards = Reaction(
        "CCl.[F-]>>CF.[Cl-]", name="sn2_forwards", solvent_name="water"
    )

    backwards = Reaction(
        "CF.[Cl-]>>CCl.[F-]", name="sn2_backwards", solvent_name="water"
    )

    reaction = MultiStepReaction(forwards, backwards)
    reaction.calculate_reaction_profile()

    assert reaction.reactions is not None
    assert len(reaction.reactions) == 2
    assert reaction.reactions[0].ts is not None


def test_balancing():
    """Test that a multistep reaction can balance using addition of
    spectator molecules in some reactions"""
    h2 = Product(atoms=[Atom("H"), Atom("H", x=0.7)])

    r1 = Reaction(Reactant(atoms=[Atom("H")]), Reactant(atoms=[Atom("H")]), h2)

    h2_he = Product(atoms=[Atom("H"), Atom("H", x=0.5), Atom("He", x=1.0)])

    r2 = Reaction(h2.to_reactant(), Reactant(atoms=[Atom("He")]), h2_he)

    # Should be able to form a multistep reaction, even if the total number of
    # atoms doesn't balance between reactions
    rxn = MultiStepReaction(r1, r2)

    assert rxn.reactions[0].atomic_symbols != rxn.reactions[1].atomic_symbols

    # Balancing should add a He atom to the reactants and products of
    # the first reaction
    rxn._balance()
    assert rxn.reactions[0].atomic_symbols == rxn.reactions[1].atomic_symbols
    assert sum(m.atomic_symbols == ["He"] for m in rxn.reactions[0].reacs) == 1
    assert sum(m.atomic_symbols == ["He"] for m in rxn.reactions[0].prods) == 1

    # New reaction where a He atom is removed i.e.
    # H2 + He -> H2.He
    # H2 -> H + H

    rev_rxn = MultiStepReaction(r2, r1)
    rev_rxn._balance()
    first_rxn, second_rxn = rev_rxn.reactions

    # Now the 2nd reaction should have additional He atoms
    assert first_rxn.has_identical_composition_as(second_rxn)
    assert sum(m.atomic_symbols == ["He"] for m in second_rxn.reacs) == 1
    assert sum(m.atomic_symbols == ["He"] for m in second_rxn.prods) == 1


def test_impossible_balance():

    with pytest.raises(ValueError):
        # No previous reaction to the first
        MultiStepReaction()._set_reactants_from_previous_products(0)

    r1 = Reaction(
        Reactant(atoms=[Atom("I")]),
        Reactant(atoms=[Atom("I")]),
        Product(atoms=[Atom("I"), Atom("I", x=2.0)]),
    )

    r2 = Reaction(
        Reactant(atoms=[Atom("Cl")]),
        Reactant(atoms=[Atom("Cl")]),
        Product(atoms=[Atom("Cl"), Atom("Cl", x=1.8)]),
    )

    rxn = MultiStepReaction(r1, r2)
    with pytest.raises(RuntimeError):
        rxn._balance()

    with pytest.raises(RuntimeError):
        rxn._set_reactants_from_previous_products(step_idx=1)

    r3 = Reaction(
        Reactant(atoms=[Atom("Cl"), Atom("Cl", x=1.8)]),
        Product(atoms=[Atom("Cl")]),
        Product(atoms=[Atom("Cl")]),
    )
    identity_rxn = MultiStepReaction(r2, r3)

    # No added molecule for an a multistep reaction with two identical rxns
    with pytest.raises(RuntimeError):
        _ = identity_rxn._added_molecule(step_idx=0, next_step_idx=1)


def test_multistep_reaction_invalid_init():

    # Must form a multistep reaction out of Reaction instances
    with pytest.raises(ValueError):
        _ = MultiStepReaction("a")
