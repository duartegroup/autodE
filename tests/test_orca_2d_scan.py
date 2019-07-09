import os
from autode import *
here = os.path.dirname(os.path.abspath(__file__))


def test_2d_scan():

    os.chdir(os.path.join(here, 'orca_2d_scan'))

    quinone_smiles = 'O=C(C=C1)C=CC1=O'
    cyclopentadiene_smiles = 'C1=CC=CC1'
    prod_smiles = 'O=C([C@]1([H])[C@@]2([H])[C@@H]3C=C[C@H]1C3)C=CC2=O'

    acrylaldehyde = Reactant(name='acrylaldehyde', smiles=quinone_smiles)
    butadiene = Reactant(name='butadiene', smiles=cyclopentadiene_smiles)
    product = Product(name='product', smiles=prod_smiles)
    reaction = Reaction(acrylaldehyde, butadiene, product)
    reaction.locate_transition_state()

    assert reaction.ts.energy == -574.486203434787
    assert len(reaction.ts.imag_freqs) == 1
    assert reaction.ts.imag_freqs[0] == -382.79
