"""
Test that an SN2' substitution reaction is correctly generated
"""
from autode.reactions import Reaction
from autode.species import ReactantComplex
from autode.atoms import Atom
from autode.reactions.reaction_types import Substitution
from autode.species import Reactant, Product
from autode.bond_rearrangement import get_bond_rearrangs
from autode.bond_rearrangement import BondRearrangement
from autode.substitution import get_substc_and_add_dummy_atoms
from autode.input_output import xyz_file_to_atoms
from autode.transition_states.locate_tss import translate_rotate_reactant
from . import testutils
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_detection():

    # F- + H2CCHCH2Cl -> FCH2CHCH2 + Cl-
    reaction = Reaction(
        Reactant(name="F-", charge=-1, atoms=[Atom("F")]),
        Reactant(name="alkeneCl", smiles="C=CCCl"),
        Product(name="alkeneF", smiles="C=CCF"),
        Product(name="Cl-", charge=-1, atoms=[Atom("Cl")]),
    )

    assert reaction.type == Substitution

    reactant, product = reaction.reactant, reaction.product

    bond_rearrs = get_bond_rearrangs(reactant, product, name="SN2")

    # autodE should find both direct SN2 and SN2' pathways
    assert len(bond_rearrs) == 2
    os.remove("SN2_BRs.txt")


@testutils.work_in_zipped_dir(os.path.join(here, "data", "sn2prime.zip"))
def test_subst():

    reactant = Reactant(name="sn2_r", atoms=xyz_file_to_atoms("reactant.xyz"))

    # SN2' bond rearrangement
    bond_rearr = BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(3, 4)]
    )

    subst_centers = get_substc_and_add_dummy_atoms(
        reactant, bond_rearr, shift_factor=1.0
    )

    assert len(subst_centers) == 1

    # get_substitution_centres should add a dummy atom so the ACX angle is
    # defined
    assert len(reactant.atoms) == 11


@testutils.work_in_zipped_dir(os.path.join(here, "data", "sn2prime.zip"))
def test_translate_rotate():

    reactant = ReactantComplex(
        Reactant(name="F-", charge=-1, atoms=[Atom("F")]),
        Reactant(name="alkeneCl", atoms=xyz_file_to_atoms("alkene.xyz")),
    )

    assert reactant.n_molecules == 2

    # Initially the geometry is not sensible
    assert reactant.distance(0, 2) < 1.0

    # SN2' bond rearrangement
    bond_rearr = BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(3, 4)]
    )

    translate_rotate_reactant(reactant, bond_rearr, shift_factor=1.5)
    assert len(reactant.atoms) == 10

    # The geometry should now be sensible
    for i in range(1, 10):
        assert reactant.distance(0, i) > 2.0

    # Should be closer to the end carbon than the middle
    assert reactant.distance(0, 1) < reactant.distance(0, 2)
