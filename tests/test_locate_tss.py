from autode.transition_states import locate_tss
from autode import molecule
from autode import reaction


def test_get_reactant_and_product_compleses():

    #h + h > h2
    h1 = molecule.Product(name='h1', xyzs=[['H', 0.0, 0.0, 0.0]])
    h2 = molecule.Product(name='h2', xyzs=[['H', 1.0, 0.0, 0.0]])
    hh = molecule.Reactant(
        name='hh', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    hh_reac = reaction.Reaction(mol1=h1, mol2=h2, mol3=hh, name='h2_assoc')

    reactant, product = locate_tss.get_reactant_and_product_complexes(hh_reac)
    assert type(reactant) == molecule.Reactant
    assert type(product) == molecule.Molecule

    methane_reactant = molecule.Reactant(name='methane1', smiles='C')
    methane_product = molecule.Product(name='methane2', smiles='C')
    cc_reac = reaction.Reaction(mol1=methane_reactant, mol2=methane_product)

    reactant2, product2 = locate_tss.get_reactant_and_product_complexes(
        cc_reac)
    assert type(reactant2) == molecule.Reactant
    assert type(product2) == molecule.Product
