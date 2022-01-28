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
    assert coords.shape == (2 * 3,)
