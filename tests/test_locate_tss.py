import os
import pytest
import autode.exceptions as ex
from autode.atoms import Atom
from autode import Reactant, Product, Reaction
from autode.species.complex import ReactantComplex, ProductComplex
from autode.reactions.reaction_types import Dissociation
from autode.bond_rearrangement import get_bond_rearrangs, BondRearrangement
from autode.transition_states.locate_tss import (
    get_ts,
    ts_guess_funcs_prms,
    find_tss,
)


def test_one_to_three_dissociation():

    r = Reactant(name="tet_int", smiles="CC(OS(Cl)=O)(Cl)O")
    p1 = Product(name="acyl", smiles="CC(Cl)=[OH+]")
    p2 = Product(name="so2", smiles="O=S=O")
    p3 = Product(name="chloride", smiles="[Cl-]")

    reaction = Reaction(r, p1, p2, p3, solvent_name="thf")
    assert reaction.type is Dissociation

    # Generate reactants and product complexes then find the single possible
    # bond rearrangement
    reactant, product = reaction.reactant, reaction.product
    bond_rearrangs = get_bond_rearrangs(reactant, product, name=str(reaction))
    assert len(bond_rearrangs) == 1
    os.remove(f"{str(reaction)}_BRs.txt")

    # This dissociation breaks two bonds and forms none
    bond_rearrangement = bond_rearrangs[0]
    assert len(bond_rearrangement.fbonds) == 0
    assert len(bond_rearrangement.bbonds) == 2

    # Ensure there is at least one bond function that could give the TS
    try:
        ts_funcs_params = ts_guess_funcs_prms(
            str(reaction), reactant, product, bond_rearrangement
        )
        assert len(list(ts_funcs_params)) > 0

    # Allow this function to be run with no avail EST methods
    except ex.MethodUnavailable:
        pass


def test_more_forming_than_breaking():

    h_a = Reactant(atoms=[Atom("H")], name="h_a")
    h_b = Reactant(atoms=[Atom("H")], name="h_b")
    h2_sep = ReactantComplex(h_a, h_b)

    h2 = ProductComplex(Product(atoms=[Atom("H"), Atom("H", x=1)], name="h2"))

    rxn = Reaction(h_a, h_b, h2)
    bond_rearr = BondRearrangement(forming_bonds=[(0, 1)], breaking_bonds=None)
    assert bond_rearr.n_fbonds > bond_rearr.n_bbonds

    # Number of bonds in the product needs to be the same or fewer than
    # the reactant currently. Will need more get_ts_guess_function_and_params
    # if this is to be supproted
    with pytest.raises(NotImplementedError):
        # name, reactant, product
        _ = get_ts(
            name=str(rxn), reactant=h2_sep, product=h2, bond_rearr=bond_rearr
        )


def test_find_tss_no_products():

    reaction = Reaction(Reactant(smiles="O"), Product(smiles="O"))

    # Check for no reactant and product before anything else
    with pytest.raises(ValueError):
        _ = find_tss(reaction)
