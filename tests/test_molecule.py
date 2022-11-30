from autode.species.molecule import Molecule
from autode.conformers import Conformer
from autode.exceptions import NoAtomsInMolecule
from autode.geom import are_coords_reasonable
from autode.input_output import atoms_to_xyz_file
from autode.smiles.smiles import calc_multiplicity, init_organic_smiles
from autode.wrappers.ORCA import orca
from autode.species.molecule import Reactant, Product
from autode.atoms import Atom
from rdkit.Chem import Mol
from . import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_basic_attributes():

    methane = Molecule(name="methane", smiles="C")

    assert methane.name == "methane"
    assert methane.smiles == "C"

    assert repr(methane) != ""  # Have some simple representation

    assert methane.energy is None
    assert methane.n_atoms == 5
    assert methane.graph.number_of_edges() == 4
    assert methane.graph.number_of_nodes() == methane.n_atoms
    assert methane.n_conformers == 0
    assert methane.charge == 0
    assert methane.mult == 1
    assert isinstance(methane.rdkit_mol_obj, Mol)

    assert np.isclose(methane.eqm_bond_distance(0, 1), 1.1, atol=0.2)  # Å

    # A molecule without a name should default to the formula
    methane = Molecule(smiles="C")
    assert methane.name == "CH4" or methane.name == "H4C"

    atoms_to_xyz_file(atoms=[Atom("H")], filename="tmp_H.xyz")
    # Cannot create a molecule with an odd number of electrons with charge = 0
    # and spin multiplicity of 1
    with pytest.raises(ValueError):
        _ = Molecule("tmp_H.xyz")

    # but is fine as a doublet
    h_atom = Molecule("tmp_H.xyz", mult=2)
    assert h_atom.mult == 2

    # or as a proton
    h_atom = Molecule("tmp_H.xyz", charge=1)
    assert h_atom.charge == 1 and h_atom.mult == 1

    os.remove("tmp_H.xyz")


def test_bond_matrix():

    water = Molecule(smiles="O")
    # check there are bonds where they are expected

    bond_matrix = water.bond_matrix

    assert not bond_matrix[0, 0]
    assert bond_matrix[0, 1]  # O-H
    assert bond_matrix[0, 2]  # O-H
    assert bond_matrix[1, 0]  # H-O
    assert bond_matrix[2, 0]  # H-O
    assert not bond_matrix[1, 2]  # H-H

    # No self bonds
    assert not bond_matrix[1, 1]
    assert not bond_matrix[2, 2]


def test_gen_conformers():

    ethane = Molecule(name="ethane", smiles="CC")
    ethane._generate_conformers(n_confs=2)

    assert ethane.rdkit_conf_gen_is_fine

    # Even though two conformers have been requested they are pruned on RMSD
    assert len(ethane.conformers) >= 1
    assert type(ethane.conformers[0]) == Conformer
    assert ethane.conformers[0].energy is None
    assert ethane.conformers[0].n_atoms == 8

    with pytest.raises(NoAtomsInMolecule):
        mol = Molecule()
        mol._generate_conformers()

    # Metal complexes must be completely bonded entities
    with pytest.raises(Exception):
        _ = Molecule(smiles="[Pd]C.C")


def test_siman_conf_gen(tmpdir):
    os.chdir(tmpdir)

    rh_complex = Molecule(
        name="[RhH(CO)3(ethene)]", smiles="O=C=[Rh]1(=C=O)(CC1)([H])=C=O"
    )
    assert are_coords_reasonable(coords=rh_complex.coordinates)
    assert rh_complex.n_atoms == 14
    assert 12 < rh_complex.graph.number_of_edges() < 15  # What is a bond even

    # Should be able to generate even crazy molecules
    mol = Molecule(smiles="C[Fe](C)(C)(C)(C)(C)(C)(C)C")
    assert mol.atoms is not None

    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "molecule.zip"))
def test_molecule_opt():

    mol = Molecule(name="H2", smiles="[H][H]")

    # Set the orca path to something that exists
    orca.path = here

    mol.optimise(method=orca)
    assert mol.energy == -1.160687049941
    assert mol.n_atoms == 2

    opt_coords = mol.coordinates
    # H2 bond length ~ 0.767 Å at PBE/def2-SVP
    assert 0.766 < np.linalg.norm(opt_coords[0] - opt_coords[1]) < 0.768


def calc_mult():

    h = Molecule(name="H", smiles="[H]")
    assert h.mult == 2

    assert calc_multiplicity(h, n_radical_electrons=1) == 2

    # Setting the multiplicity manually should override the number of radical
    # electrons derived from the SMILES string
    # note: H with M=3 is obviously not possible
    h.mult = 3
    assert calc_multiplicity(h, n_radical_electrons=1) == 3

    # Diradicals should default to singlets..
    assert calc_multiplicity(h, n_radical_electrons=2) == 1


def test_reactant_to_product_and_visa_versa():

    prod = Reactant().to_product()
    assert type(prod) is Product

    reac = Product().to_reactant()
    assert type(reac) is Reactant


@testutils.work_in_zipped_dir(os.path.join(here, "data", "molecule.zip"))
def test_molecule_from_xyz():

    h2 = Molecule("h2_conf0.xyz")
    assert h2.name == "h2_conf0"
    assert h2.n_atoms == 2
    assert h2.formula == "H2"

    # Molecules loaded from .xyz directly can still have names
    h2_named = Molecule("h2_conf0.xyz", name="tmp")
    assert h2_named.name == "tmp"

    # Name kwarg takes priority even if arg is defined
    h2_named2 = Molecule("tmp", name="tmp2")
    assert h2_named2.name == "tmp2"


def test_rdkit_possible_fail():
    """RDKit can't generate structures for some SMILES, make sure they can
    be generated in other ways"""

    rh_complex = Molecule(smiles="C[Rh](=C=O)(=C=O)(=C=O)=C=O")
    assert are_coords_reasonable(coords=rh_complex.coordinates)

    # Trying to parse with RDKit should revert to RR structure
    rh_complex_rdkit_attempt = Molecule()
    init_organic_smiles(
        rh_complex_rdkit_attempt, smiles="O=[Rh]([H])([H])([H])=O"
    )
    assert are_coords_reasonable(coords=rh_complex.coordinates)

    # RDKit also may not parse CH5+
    ch5 = Molecule(smiles="[H]C([H])([H])([H])[H+]")
    assert are_coords_reasonable(coords=ch5.coordinates)


def test_multi_ring_smiles_init():

    cr_complex = Molecule(
        smiles="[N+]12=CC=CC3=C1C(C(C=C3)=CC=C4)=[N+]4[Cr]25"
        "6([N+]7=CC=CC8=C7C9=[N+]6C=CC=C9C=C8)[N+]%10"
        "=CC=CC%11=C%10C%12=[N+]5C=CC=C%12C=C%11"
    )

    assert are_coords_reasonable(cr_complex.coordinates)


def test_prune_diff_graphs():

    h2 = Molecule(smiles="[H][H]")
    h2.energy = -1.0
    assert h2.graph is not None

    h2_not_bonded = Molecule(atoms=[Atom("H"), Atom("H", x=10)])
    h2_not_bonded.energy = -1.1
    # no bonds for a long H-bond
    assert np.allclose(h2_not_bonded.bond_matrix, np.zeros(shape=(2, 2)))

    h2.conformers = [h2_not_bonded]

    h2.conformers.prune_diff_graph(graph=h2.graph)

    # Should prune all conformers
    assert h2.n_conformers == 0


def test_lowest_energy_conformer_set_ok():

    h2 = Molecule(smiles="[H][H]")
    h2.energy = -1.0

    h2_long = Molecule(atoms=[Atom("H"), Atom("H", x=0.5)])
    h2_long.energy = -1.1

    h2.conformers = [h2_long]

    # Setting the lowest energy conformer should override the atoms
    # and energy of the molecule
    h2._set_lowest_energy_conformer()

    assert h2.energy == -1.1
    assert np.isclose(h2.distance(0, 1).to("ang"), 0.5)


def test_lowest_energy_conformer_set_no_energy():

    h2 = Molecule(smiles="[H][H]")
    h2.energy = -1.0

    # No lowest energy conformer without any conformers..
    assert h2.conformers.lowest_energy is None

    h2_conf = Molecule(atoms=[Atom("H"), Atom("H", x=1)])
    assert h2_conf.energy is None

    h2.conformers = [h2_conf]
    assert h2.conformers.lowest_energy is None

    # Cannot set the lowest energy with no conformers having defined energies
    with pytest.raises(Exception):
        h2_conf._set_lowest_energy_conformer()


def test_defined_metal_spin_state():

    mol = Molecule(smiles="[Sc]C", mult=3)
    assert mol.mult == 3


def test_atom_class_defined_for_organic():

    mol = Molecule(smiles="[Br-:1]")
    assert mol.atoms[0].atom_class is not None


def test_smiles_and_user_defined_charge_raises_exception():

    with pytest.raises(Exception):
        _ = Molecule(smiles="[Cl-]", charge=1)


def test_user_defined_charge_overrides_smiles_mult():

    ch2 = Molecule(smiles="[H][C][H]")
    default_mult = ch2.mult

    ch2_alt = Molecule(smiles="[H][C][H]", mult=3 if default_mult == 1 else 1)
    assert ch2_alt.mult != default_mult
