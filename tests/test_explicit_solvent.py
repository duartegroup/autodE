from autode.solvent import explicit_solvent
from autode.molecule import SolvatedMolecule
from autode.atoms import Atom
import numpy as np


def test_add_solvent_mols():
    species = SolvatedMolecule(atoms=[Atom('C', 0.0, 0.0, 0.0)])
    species.solvent_mol = SolvatedMolecule(atoms=[Atom('H', 0.0, 0.0, 0.0), Atom('O', 0.7, 0.0, 0.0)])
    explicit_solvent.add_solvent_molecules(species, 5, 10)

    assert len(species.qm_solvent_atoms) == 10
    assert len(species.mm_solvent_atoms) == 10
    all_atoms = species.qm_solvent_atoms + species.mm_solvent_atoms
    assert all(0.699 < np.linalg.norm(all_atoms[i*2].coord - all_atoms[i*2+1].coord) < 0.701 for i in range(10))


def test_centre_species():
    species = SolvatedMolecule(atoms=[Atom('H', 1.0, 1.0, 1.0)])
    explicit_solvent.centre_species(species)
    assert np.allclose(species.atoms[0].coord, [0, 0, 0])
