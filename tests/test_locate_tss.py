from autode import Reactant, Product, Reaction
from autode.reactions import Dissociation
from autode.complex import get_complexes
from autode.bond_rearrangement import get_bond_rearrangs
from autode.transition_states.locate_tss import get_ts_guess_function_and_params


def test_one_to_three_dissociation():

    r = Reactant(name='tet_int', smiles='CC(OS(Cl)=O)(Cl)O')
    p1 = Product(name='acyl', smiles='CC(Cl)=[OH+]')
    p2 = Product(name='so2', smiles='O=S=O')
    p3 = Product(name='chloride', smiles='[Cl-]')

    reaction = Reaction(r, p1, p2, p3, solvent_name='thf')
    assert reaction.type is Dissociation

    # Generate reactants and product complexes then find the single possible bond rearrangement
    reactant, product = get_complexes(reaction)
    bond_rearrangs = get_bond_rearrangs(reactant, product, name=str(reaction))
    assert len(bond_rearrangs) == 1

    # This dissociation breaks two bonds and forms none
    bond_rearrangement = bond_rearrangs[0]
    assert len(bond_rearrangement.fbonds) == 0
    assert len(bond_rearrangement.bbonds) == 2

    # Ensure there is at least one bond function that could give the TS
    ts_funcs_params = get_ts_guess_function_and_params(reaction, reactant, product, bond_rearrangement)
    assert len(list(ts_funcs_params)) > 0
