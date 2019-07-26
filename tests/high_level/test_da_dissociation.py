import os
from autode import *
here = os.path.dirname(os.path.abspath(__file__))


def test_da_subst():

    os.chdir(os.path.join(here, 'da_dissociation'))

    benzoquinone = Reactant(name='benzoquinone', smiles='O=C(C=C1)C=CC1=O')
    cyclopentadiene = Reactant(name='cyclopentadiene', smiles='C1=CC=CC1')
    product = Product(name='product', smiles='O=C([C@]1([H])[C@@]2([H])[C@@H]3C=C[C@H]1C3)C=CC2=O')
    reaction = Reaction(benzoquinone, cyclopentadiene, product)
    reaction.locate_transition_state()

    assert reaction.ts.energy == -574.486151149915
    assert len(reaction.ts.imag_freqs) == 1
    assert reaction.ts.imag_freqs[0] == -389.0
