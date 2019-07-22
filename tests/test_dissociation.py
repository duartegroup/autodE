import os
from autode import *
here = os.path.dirname(os.path.abspath(__file__))


def test_dissociation():

    os.chdir(os.path.join(here, 'dissociation'))

    formaldehyde = Reactant(name='formaldehyde', smiles='CC(C)=O', solvent='water')
    cn = Reactant(name='cyanide', smiles='N#[C-]', solvent='water')
    prod = Product(name='product', smiles='CC([O-])(C#N)C', solvent='water')
    reaction = Reaction(formaldehyde, cn, prod)
    reaction.locate_transition_state()
    assert -245 < reaction.ts.imag_freqs[0] < -230


if __name__ == '__main__':
    test_dissociation()
