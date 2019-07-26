import os
from autode import *
here = os.path.dirname(os.path.abspath(__file__))


def test_rearrangement():

    os.chdir(os.path.join(here, 'rearrangement'))

    reac = Reactant(name='cope_reac', smiles='C=CC(O)CC=C')
    prod = Product(name='cope_prod', smiles='C=CCC/C=C/O')
    reaction = Reaction(reac, prod)
    reaction.locate_transition_state()
    assert reaction.tss[0].imag_freqs == [-506.63]
