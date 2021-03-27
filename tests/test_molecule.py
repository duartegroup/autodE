from autode.species.molecule import Molecule
from autode.conformers.conformer import Conformer
from autode.exceptions import NoAtomsInMolecule
from autode.geom import are_coords_reasonable
from autode.input_output import atoms_to_xyz_file
from autode.smiles.smiles import calc_multiplicity, init_organic_smiles
from autode.wrappers.ORCA import orca
from autode.species.molecule import Reactant
from autode.species.molecule import Product
from autode.species.molecule import reactant_to_product
from autode.atoms import Atom
from rdkit.Chem import Mol
from . import testutils
import numpy as np
import pytest
import os

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

    # A molecule without a name should default to the formula
    methane = Molecule(smiles='C')
    assert methane.name == 'CH4' or methane.name == 'H4C'

    atoms_to_xyz_file(atoms=[Atom('H')], filename='tmp_H.xyz')
    # Cannot create a molecule with an odd number of electrons with charge = 0
    # and spin multiplicity of 1
    with pytest.raises(ValueError):
        _ = Molecule('tmp_H.xyz')

    # but is fine as a doublet
    h_atom = Molecule('tmp_H.xyz', mult=2)
    assert h_atom.mult == 2

    os.remove('tmp_H.xyz')


def test_gen_conformers():

    ethane = Molecule(name='ethane', smiles='CC')
    ethane._generate_conformers(n_confs=2)

    assert ethane.rdkit_conf_gen_is_fine
    assert type(ethane.conformers) == list

    # Even though two conformers have been requested they are pruned on RMSD
    assert len(ethane.conformers) >= 1
    assert type(ethane.conformers[0]) == Conformer
    assert ethane.conformers[0].energy is None
    assert ethane.conformers[0].n_atoms == 8

    with pytest.raises(NoAtomsInMolecule):
        mol = Molecule()
        mol._generate_conformers()

    # Metal complexes must be completely bonded entities
    with pytest.raises(ValueError):
        _ = Molecule(smiles='[Pd]C.C')


def test_siman_conf_gen(tmpdir):
    os.chdir(tmpdir)

    rh_complex = Molecule(name='[RhH(CO)3(ethene)]',
                          smiles='O=C=[Rh]1(=C=O)(CC1)([H])=C=O')
    assert are_coords_reasonable(coords=rh_complex.coordinates)
    assert rh_complex.n_atoms == 14
    assert 12 < rh_complex.graph.number_of_edges() < 15  # What is a bond even

    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'molecule.zip'))
def test_molecule_opt():

    mol = Molecule(name='H2', smiles='[H][H]')

    # Set the orca path to something that exists
    orca.path = here

    mol.optimise(method=orca)
    assert mol.energy == -1.160687049941
    assert mol.n_atoms == 2

    opt_coords = mol.coordinates
    # H2 bond length ~ 0.767 Å at PBE/def2-SVP
    assert 0.766 < np.linalg.norm(opt_coords[0] - opt_coords[1]) < 0.768


def calc_mult():

    h = Molecule(name='H', smiles='[H]')
    assert calc_multiplicity(h, n_radical_electrons=1) == 2

    # Setting the multiplicity manually should override the number of radical
    # electrons derived from the SMILES string
    # note: H with M=3 is obviously not possible
    h.mult = 3
    assert calc_multiplicity(h, n_radical_electrons=1) == 3

    # Diradicals should default to singlets..
    assert calc_multiplicity(h, n_radical_electrons=2) == 1


def test_reactant_to_product():

    methane = Reactant(smiles='C', charge=0, mult=1)
    prod = reactant_to_product(reactant=methane)

    assert type(prod) is Product


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'molecule.zip'))
def test_molecule_from_xyz():

    h2 = Molecule('h2_conf0.xyz')
    assert h2.name == 'h2_conf0'
    assert h2.n_atoms == 2
    assert h2.formula() == 'H2'


def test_rdkit_possible_fail():
    """RDKit can't generate structures for some SMILES, make sure they can
    be generated in other ways"""

    rh_complex = Molecule(smiles='C[Rh](=C=O)(=C=O)(=C=O)=C=O')
    assert are_coords_reasonable(coords=rh_complex.coordinates)

    # Trying to parse with RDKit should revert to RR structure
    rh_complex_rdkit_attempt = Molecule()
    init_organic_smiles(rh_complex_rdkit_attempt,
                        smiles='O=[Rh]([H])([H])([H])=O')
    assert are_coords_reasonable(coords=rh_complex.coordinates)

    # RDKit also may not parse CH5+
    ch5 = Molecule(smiles='[H]C([H])([H])([H])[H+]')
    assert are_coords_reasonable(coords=ch5.coordinates)
