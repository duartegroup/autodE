from autode import molecule
from autode import confomers
from rdkit.Chem import Mol
import numpy as np
import os
here = os.path.dirname(os.path.abspath(__file__))


def test_basic_attributes():

    methane = molecule.Molecule(name='methane', smiles='C')

    assert methane.name == 'methane'
    assert methane.smiles == 'C'
    assert methane.energy is None
    assert methane.n_atoms == 5
    assert methane.n_bonds == 4
    assert methane.graph.number_of_edges() == 4
    assert methane.graph.number_of_nodes() == methane.n_atoms
    assert methane.conformers is None
    assert methane.charge == 0
    assert methane.mult == 1
    assert methane.distance_matrix.shape == (5, 5)
    assert -0.001 < np.trace(methane.distance_matrix) < 0.001
    assert isinstance(methane.mol_obj, Mol)

    h2 = molecule.Molecule(name='h2', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])

    assert h2.n_atoms == 2
    assert h2.distance_matrix.shape == (2, 2)
    assert h2.smiles is None
    assert h2.graph.number_of_edges() == 1
    assert h2.graph.number_of_nodes() == h2.n_atoms


def test_rdkit_conf_generation():

    h2 = molecule.Molecule(name='h2', smiles='[H][H]')
    h2.check_rdkit_graph_agreement()

    h2.generate_rdkit_conformers(n_rdkit_confs=1)
    assert isinstance(h2.conformers[0], confomers.Conformer)
    assert len(h2.conformers) == 1
    assert h2.n_conformers == 1


def test_xtb_conf_optimisation_and_strip():

    os.chdir(os.path.join(here, 'conformers'))

    methane = molecule.Molecule(name='methane', smiles='C')
    methane.generate_rdkit_conformers(n_rdkit_confs=50)
    methane.optimise_conformers_xtb()
    methane.strip_non_unique_confs()
    assert methane.n_conformers == 1
