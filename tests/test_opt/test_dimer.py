import numpy as np
import pytest
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.opt.coordinates.dimer import DimerCoordinates


def test_dimer_coord_init():

    mol1 = Molecule()
    mol2 = Molecule(atoms=[Atom('H')])

    # Dimer coordinates must be created from two species with the same
    # atomic composition
    with pytest.raises(ValueError):
        _ = DimerCoordinates.from_species(mol1, mol2)

    # Dimer coordinates are concatenated cartesian coordinates
    coords = DimerCoordinates.from_species(mol2, mol2)
    assert coords.shape == (3, 3)

    assert coords.g is None
    assert coords.h is None


def test_dimer_coord_init_polyatomic():

    mol1 = Molecule(atoms=[Atom('H'), Atom('H', x=1.0)])
    mol2 = Molecule(atoms=[Atom('H', 0.1), Atom('H', x=1.1)])

    coords = DimerCoordinates.from_species(mol1, mol2)
    assert coords.shape == (3, 6)

    assert np.allclose(coords.x0,
                       np.array([0.05, 0., 0., 1.05, 0., 0.]))

    assert np.allclose(coords.x1,
                       np.array([0., 0., 0., 1., 0., 0.]))

    assert np.allclose(coords.x2,
                       np.array([0.1, 0., 0., 1.1, 0., 0.]))

    # Gradient has not been evaluated
    with pytest.raises(Exception):
        _ = coords.g0
