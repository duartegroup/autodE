import os
from autode import *
here = os.path.dirname(os.path.abspath(__file__))


def test_substitution():

    os.chdir(os.path.join(here, 'substitution2'))

    reac1 = Reactant(name='NH3', smiles='[H]N([H])[H]')
    reac2 = Reactant(name='CH3NH3+', smiles='[H][N+]([H])(C)[H]')
    prod1 = Product(name='NH3', smiles='[H]N([H])[H]')
    prod2 = Product(name='CH3NH3+', smiles='[H][N+]([H])(C)[H]')
    reaction = Reaction(reac1, reac2, prod1, prod2)
    reaction.locate_transition_state()

    assert reaction.ts.imag_freqs == [-537.39]
