import os
from autode import *
here = os.path.dirname(os.path.abspath(__file__))


def test_substitution():

    os.chdir(os.path.join(here, 'substitution'))

    reac1 = Reactant(name='F-', xyzs=[['F', 0.0, 0.0, 0.0]], charge=-1, solvent='water')        # RDKit is slow with
    reac2 = Reactant(name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent='water')                  # single atom molecules
    prod1 = Product(name='Cl-', xyzs=[['Cl', 0.0, 0.0, 0.0]], charge=-1, solvent='water')       # so init with xyzs
    prod2 = Product(name='CH3F', smiles='[H]C([H])(F)[H]', solvent='water')
    reaction = Reaction(reac1, reac2, prod1, prod2)
    reaction.locate_transition_state()

    assert reaction.ts.imag_freqs == [-509.63]
