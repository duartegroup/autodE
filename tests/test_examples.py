"""
Tests that are in the documentation. If they break the documentation is wrong!
If there is any change to the code please also change the examples to
accommodate the changes.
"""
from autode.species import Species
from autode.atoms import Atom
import numpy as np


def test_manipulation():

    species = Species(name='species', atoms=None, charge=0, mult=1)
    assert species.n_atoms == 0

    h2 = Species(name='H2', charge=0, mult=1, atoms=[Atom('H'), Atom('H')])
    assert h2.n_atoms == 2

    # Expecting both atoms to be initialised at the origin
    assert np.linalg.norm(h2.atoms[0].coord - h2.atoms[1].coord) < 1E-6

    atom1, atom2 = h2.atoms
    atom1.translate(vec=np.array([1.0, 0.0, 0.0]))
    atom1.rotate(theta=np.pi, axis=np.array([0.0, 0.0, 1.0]))

    assert np.linalg.norm(atom1.coord - np.array([-1., 0., 0.])) < 1E-6

    assert h2.solvent is None

    f = Species(name='F-', charge=-1, mult=1, atoms=[Atom('F')], solvent_name='DCM')
    assert f.solvent.g09 == 'Dichloromethane'
    assert f.solvent.xtb == 'CH2Cl2'
