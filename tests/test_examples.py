"""
Tests that are in the documentation. If they break the documentation is wrong!
If there is any change to the code please also change the examples to
accommodate the changes.
"""
from autode.species import Species
from autode.species import Molecule
from autode.atoms import Atom
import numpy as np


def test_species():

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


def test_molecule():

    molecule = Molecule(name='molecule')
    assert molecule.charge == 0
    assert molecule.mult == 1

    water = Molecule(name='h2o', smiles='O')
    assert water.n_atoms == 3
    assert all(node in water.graph.nodes for node in (0, 1, 2))
    assert (0, 1) in water.graph.edges
    assert (0, 2) in water.graph.edges

    # Shift so the first atom is at the origin
    water.translate(vec=-water.atoms[0].coord)
    assert np.linalg.norm(water.atoms[0].coord - np.zeros(3)) < 1E-6

