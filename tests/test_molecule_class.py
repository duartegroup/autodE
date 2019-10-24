from autode.molecule import Molecule
from autode.molecule import Reactant
from autode.molecule import Product
from autode.reaction import Reaction
from autode.conformers import conformers
from rdkit.Chem import Mol
import numpy as np
import pytest
from autode.transition_states.locate_tss import get_reactant_and_product_complexes


h2 = Molecule(name='h2', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])


def test_basic_attributes():

    methane = Molecule(name='methane', smiles='C')

    assert methane._calc_multiplicity(1) == 2
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

    assert h2.n_atoms == 2
    assert h2.distance_matrix.shape == (2, 2)
    assert h2.smiles is None
    assert h2.graph.number_of_edges() == 1
    assert h2.graph.number_of_nodes() == h2.n_atoms


def test_rdkit_conf_generation():

    h2 = Molecule(name='h2', smiles='[H][H]')
    h2._check_rdkit_graph_agreement()

    h2.generate_conformers(n_rdkit_confs=1)
    assert isinstance(h2.conformers[0], conformers.Conformer)
    assert len(h2.conformers) == 1
    assert h2.n_conformers == 1


def test_attributes_methods():

    h_atom = Molecule(name='h', xyzs=[['H', 0.0, 0.0, 0.0]], mult=2)
    assert h_atom.mult == 2
    assert h2._calc_multiplicity(n_radical_electrons=2) == 1

    h2_rdkit = Molecule(name='h2', smiles='[H][H]')
    h2_rdkit.graph.remove_edge(0, 1)

    with pytest.raises(SystemExit):
        h2_rdkit._check_rdkit_graph_agreement()

    assert 0.69 < h2.calc_bond_distance(bond=(0, 1)) < 0.71
    assert len(h2.get_possible_forming_bonds()
               ) == 0        # Can't form any bonds

    assert h2.get_bonded_atoms_to_i(atom_i=0) == [1]

    h2.set_xyzs(xyzs=None)
    assert h2.xyzs is None
    h2.set_xyzs(xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.7, 0.0, 0.0]])
    assert h2.n_bonds == 1
    assert h2.n_atoms == 2


def test_conf_strip():

    # Pop[ulate a list with two identical conformers with the same energy for H2
    h2.conformers = [conformers.Conformer(name='c1', energy=-0.5),
                     conformers.Conformer(name='c1', energy=-0.5)]
    h2.n_conformers = 2

    h2.strip_non_unique_confs(energy_threshold_kj=1)
    assert h2.n_conformers == 1


def test_get_core():
    reac1 = Reactant(smiles='BrC(C1CC(C)CCC1)CC(C)C')
    reac2 = Reactant(smiles='[OH-]')
    prod1 = Product(smiles='OC(C1CC(C)CCC1)CC(C)C')
    prod2 = Product(xyzs=[['Br', 0.0, 0.0, 0.0]], charge=-1)
    reaction = Reaction(reac1, reac2, prod1, prod2)
    reactant, product = get_reactant_and_product_complexes(reaction)

    fragment_1 = reactant.strip_core()
    assert reactant.xyzs == fragment_1.xyzs
    assert reactant.stripped == False

    reactant.active_atoms = [0, 1, 36]

    fragment_2 = reactant.strip_core()
    assert len(fragment_2.xyzs) == 29
    assert reactant.stripped == True
