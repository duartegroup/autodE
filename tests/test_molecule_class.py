from autode.molecule import Molecule
from autode.conformers.conformers import Conformer
from autode.exceptions import NoAtomsInMolecule
from autode.molecule import Reactant
from autode.molecule import Product
from autode.reaction import Reaction
from autode.conformers import conformers
from autode.bond_rearrangement import BondRearrangement
from rdkit.Chem import Mol
import numpy as np
import pytest
from autode.transition_states.locate_tss import get_reactant_and_product_complexes
from autode.mol_graphs import reac_graph_to_prods


def test_basic_attributes():

    with pytest.raises(NoAtomsInMolecule):
        Molecule(atoms=[])
        Molecule()

    methane = Molecule(name='methane', smiles='C')

    assert methane._calc_multiplicity(1) == 2
    assert methane.name == 'methane'
    assert methane.smiles == 'C'
    assert methane.energy is None
    assert methane.n_atoms == 5
    assert methane.graph.number_of_edges() == 4
    assert methane.graph.number_of_nodes() == methane.n_atoms
    assert methane.conformers is None
    assert methane.charge == 0
    assert methane.mult == 1
    assert isinstance(methane.rdkit_mol_obj, Mol)


def test_gen_conformers():

    ethane = Molecule(name='ethane', smiles='CC')
    ethane._generate_conformers(n_rdkit_confs=2)

    assert ethane.rdkit_conf_gen_is_fine
    assert type(ethane.conformers) == list
    assert len(ethane.conformers) >= 1          # Even though two conformers have been requested they are pruned on RMSD
    assert type(ethane.conformers[0]) == Conformer
    assert ethane.conformers[0].energy is None
    assert ethane.conformers[0].n_atoms == 8

    with pytest.raises(NoAtomsInMolecule):
        mol = Molecule()
        mol._generate_conformers()



"""
def test_rdkit_conf_generation():

    h2 = Molecule(name='mol', smiles='[H][H]')

    h2._generate_conformers(n_rdkit_confs=1)
    assert isinstance(h2.conformers[0], conformers.Conformer)
    assert len(h2.conformers) == 1
    assert h2.n_conformers == 1


def test_attributes_methods():

    h_atom = Molecule(name='h', xyzs=[['H', 0.0, 0.0, 0.0]], mult=2)
    assert h_atom.mult == 2
    assert h2._calc_multiplicity(n_radical_electrons=2) == 1

    h2_rdkit = Molecule(name='mol', smiles='[H][H]')
    h2_rdkit.graph.remove_edge(0, 1)

    assert 0.69 < h2.calc_bond_distance(bond=(0, 1)) < 0.71

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


def test_set_pi_bonds():
    mol = Molecule(smiles='C=CC')
    assert mol.pi_bonds == [(0, 1)]

    mol2 = Molecule(smiles='C=CC=C')
    assert mol2.pi_bonds == [(0, 1), (2, 3)]


def test_stripping_core():
    reac1 = Reactant(smiles='BrC(CC(C)C)C1CC1CC')
    reac2 = Reactant(smiles='[OH-]')
    prod1 = Product(smiles='OC(CC(C)C)C1CC1CC')
    prod2 = Product(xyzs=[['Br', 0.0, 0.0, 0.0]], charge=-1)
    reaction = Reaction(reac1, reac2, prod1, prod2)
    for mol in reaction.reacs + reaction.prods:
        mol.charges = []
    reactant, _ = get_reactant_and_product_complexes(reaction)
    bond_rearrang = BondRearrangement(forming_bonds=[(1, 30)], breaking_bonds=[(0, 1)])
    product_graph = reac_graph_to_prods(reactant.graph, bond_rearrang)
    # test get core atoms
    assert reactant.get_core_atoms(product_graph) is None

    reactant.active_atoms = [0, 1, 30]
    core1 = reactant.get_core_atoms(product_graph, depth=2)
    assert len(core1) == 15

    core2 = reactant.get_core_atoms(product_graph, depth=3)
    assert len(core2) == 17

    core3 = reactant.get_core_atoms(product_graph, depth=5)
    assert len(core3) == 32

    reactant.pi_bonds = [[2, 3]]
    core4 = reactant.get_core_atoms(product_graph, depth=2)
    assert len(core4) == 17

    # test strip core
    fragment_1, bond_rearrang_1 = reactant.strip_core(core_atoms=None, bond_rearrang=bond_rearrang)
    assert reactant == fragment_1
    assert fragment_1.is_fragment == False
    assert bond_rearrang_1 == bond_rearrang

    fragment_2, bond_rearrang_2 = reactant.strip_core(core_atoms=core1, bond_rearrang=bond_rearrang)
    assert len(fragment_2.xyzs) == 17
    assert fragment_2.is_fragment == True
    assert bond_rearrang_2.fbonds == [(1, 13)]
    assert bond_rearrang_2.bbonds == [(0, 1)]

    fragment_3, bond_rearrang_3 = reactant.strip_core(core_atoms=core3, bond_rearrang=bond_rearrang)
    assert reactant == fragment_3
    assert bond_rearrang_3 == bond_rearrang
"""