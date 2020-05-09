from autode.molecule import Molecule
from autode.conformers.conformer import Conformer
from autode.exceptions import NoAtomsInMolecule
from autode.geom import are_coords_reasonable
from autode.smiles import calc_multiplicity
from autode.wrappers.ORCA import orca
from autode.molecule import Reactant
from autode.molecule import Product
from autode.molecule import reactant_to_product
from autode.reaction import Reaction
from autode.conformers import conformers
from autode.bond_rearrangement import BondRearrangement
from rdkit.Chem import Mol
import numpy as np
import pytest
import os
from autode.mol_graphs import reac_graph_to_prod_graph
here = os.path.dirname(os.path.abspath(__file__))


def test_basic_attributes():

    with pytest.raises(NoAtomsInMolecule):
        Molecule(atoms=[])
        Molecule()

    methane = Molecule(name='methane', smiles='C')

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
    ethane._generate_conformers(n_confs=2)

    assert ethane.rdkit_conf_gen_is_fine
    assert type(ethane.conformers) == list
    assert len(ethane.conformers) >= 1          # Even though two conformers have been requested they are pruned on RMSD
    assert type(ethane.conformers[0]) == Conformer
    assert ethane.conformers[0].energy is None
    assert ethane.conformers[0].n_atoms == 8

    with pytest.raises(NoAtomsInMolecule):
        mol = Molecule()
        mol._generate_conformers()


def test_siman_conf_gen(tmpdir):
    os.chdir(tmpdir)

    rh_complex = Molecule(name='[RhH(CO)3(ethene)]', smiles='O=C=[Rh]1(=C=O)(CC1)([H])=C=O')
    assert are_coords_reasonable(coords=rh_complex.get_coordinates())
    assert rh_complex.n_atoms == 14
    assert 12 < rh_complex.graph.number_of_edges() < 15     # What is a bond even

    os.chdir(here)


def test_molecule_opt():

    os.chdir(os.path.join(here, 'data'))

    mol = Molecule(name='H2', smiles='[H][H]')

    # Set the orca path to something that exists
    orca.path = here

    mol.optimise(method=orca)
    assert mol.energy == -1.160687049941
    assert mol.n_atoms == 2

    opt_coords = mol.get_coordinates()
    assert 0.766 < np.linalg.norm(opt_coords[0] - opt_coords[1]) < 0.768      # H2 bond length ~ 0.767 Å at PBE/def2-SVP

    os.remove('H2_opt_orca.inp')
    os.remove('H2_optimised_orca.xyz')
    os.chdir(here)


def calc_mult():

    h = Molecule(name='H', smiles='[H]')
    assert calc_multiplicity(h, n_radical_electrons=1) == 2

    # Setting the multiplicity manually should override the number of radical electrons derived from the SMILES string
    # note: H with M=3 is obviously not possible
    h.mult = 3
    assert calc_multiplicity(h, n_radical_electrons=1) == 3

    # Diradicals should default to singlets..
    assert calc_multiplicity(h, n_radical_electrons=2) == 1


def test_reactant_to_product():

    methane = Reactant(smiles='C', charge=0, mult=1)
    prod = reactant_to_product(reactant=methane)

    assert type(prod) is Product
