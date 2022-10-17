from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.transition_states.ts_guess import TSguess


def test_that_a_molecules_solvent_is_inherited():

    mol = Molecule(atoms=[Atom("H")], mult=2, solvent_name="water")
    assert mol.solvent.smiles == "O"

    ts_guess = TSguess.from_species(mol)
    assert ts_guess.solvent.smiles == "O"
