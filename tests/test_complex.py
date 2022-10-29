import shutil
from autode.exceptions import NoConformers
from autode.species.complex import Complex, NCIComplex
from autode.config import Config
from autode.methods import XTB
from autode.species.molecule import Molecule
from autode.geom import are_coords_reasonable
from autode.atoms import Atom
from autode.values import Distance
from autode.utils import work_in_tmp_dir
import numpy as np
from . import testutils
from copy import deepcopy
import pytest

h1 = Atom(atomic_symbol="H", x=0.0, y=0.0, z=0.0)
h2 = Atom(atomic_symbol="H", x=0.0, y=0.0, z=1.0)

hydrogen = Molecule(name="H2", atoms=[h1, h2], charge=0, mult=1)
h = Molecule(name="H", atoms=[h1], charge=0, mult=2)

monomer = Complex(hydrogen)
dimer = Complex(hydrogen, hydrogen)
trimer = Complex(hydrogen, hydrogen, hydrogen)

h2_h = Complex(hydrogen, h)
h_h = Complex(h, h)


def test_complex_class():

    blank_complex = Complex()
    assert blank_complex.n_molecules == 0
    assert blank_complex.solvent is None
    assert blank_complex.atoms is None
    assert blank_complex != "a"

    assert monomer.charge == 0
    assert monomer.mult == 1
    assert monomer.n_atoms == 2

    assert repr(monomer) != ""  # Have some simple representation

    assert h2_h.charge == 0
    assert h2_h.mult == 2
    assert h2_h.n_atoms == 3

    assert h_h.mult == 3

    assert trimer.n_atoms == 6

    # Cannot have a complex in a different solvent
    with pytest.raises(AssertionError):
        h2_water = Molecule(
            name="H2", atoms=[h1, h2], charge=0, mult=1, solvent_name="water"
        )
        _ = Complex(hydrogen, h2_water)

    # Test solvent setting
    dimer_solv = Complex(hydrogen, hydrogen, solvent_name="water")
    assert dimer_solv.solvent is not None
    assert dimer_solv.solvent.name == "water"


def test_complex_class_set():

    h2_complex = Complex(hydrogen, hydrogen, copy=True)
    assert h2_complex.charge == 0
    assert h2_complex.mult == 1

    # Cannot set the atoms of a (H2)2 complex with a single atom
    with pytest.raises(ValueError):
        h2_complex.atoms = [Atom("H")]

    with pytest.raises(ValueError):
        h2_complex.atoms = [Atom("H"), Atom("H"), Atom("H")]

    with pytest.raises(ValueError):
        h2_complex.atoms = [
            Atom("H"),
            Atom("H"),
            Atom("H"),
            Atom("H"),
            Atom("H"),
        ]

    # but can with 4 atoms
    h2_complex.atoms = [Atom("H"), Atom("H"), Atom("H"), Atom("H", x=10.0)]
    assert h2_complex.n_atoms == 4
    assert h2_complex.n_molecules == 2
    assert h2_complex.distance(0, 3) == Distance(10.0, units="ang")

    # Setting no atoms should clear the complex
    h2_complex.atoms = None
    assert h2_complex.n_molecules == 0


def test_translation():

    # Monomer translation
    monomer_copy = deepcopy(monomer)
    monomer_copy.translate_mol(vec=np.array([1.0, 0.0, 0.0]), mol_index=0)

    assert (
        np.linalg.norm(monomer_copy.atoms[0].coord - np.array([1.0, 0.0, 0.0]))
        < 1e-9
    )
    assert (
        np.linalg.norm(monomer_copy.atoms[1].coord - np.array([1.0, 0.0, 1.0]))
        < 1e-9
    )

    # Dimer translation
    dimer_copy = deepcopy(dimer)
    dimer_copy.translate_mol(vec=np.array([1.0, 0.0, 0.0]), mol_index=0)

    assert (
        np.linalg.norm(dimer_copy.atoms[0].coord - np.array([1.0, 0.0, 0.0]))
        < 1e-9
    )
    assert (
        np.linalg.norm(dimer_copy.atoms[1].coord - np.array([1.0, 0.0, 1.0]))
        < 1e-9
    )

    # Cannot translate molecule index 2 in a complex with only 2 molecules
    with pytest.raises(Exception):
        dimer_copy.translate_mol(vec=np.array([1.0, 0.0, 0.0]), mol_index=2)


def test_rotation():

    dimer_copy = deepcopy(dimer)
    with pytest.raises(Exception):
        dimer_copy.rotate_mol(mol_index=3, axis=[1.0, 1.0, 1.0], theta=0)

    dimer_copy.rotate_mol(
        axis=np.array([1.0, 0.0, 0.0]),
        theta=np.pi,
        origin=np.array([0.0, 0.0, 0.0]),
        mol_index=0,
    )

    expected_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    assert np.sum(expected_coords - dimer_copy.coordinates[[0, 1], :]) < 1e-9


def test_graph():

    hydrogen2 = deepcopy(hydrogen)
    hydrogen2.translate(vec=np.array([10, 0, 0]))

    dimer_shifted = Complex(hydrogen, hydrogen2)
    assert hasattr(dimer_shifted, "graph")
    assert dimer_shifted.graph.number_of_edges() == 0
    assert dimer_shifted.graph.number_of_nodes() == 4


def test_init_geometry():

    water = Molecule(smiles="O")
    assert are_coords_reasonable(coords=Complex(water).coordinates)

    water_dimer = Complex(water, water, do_init_translation=True)
    # water_dimer.print_xyz_file(filename='tmp.xyz')
    assert are_coords_reasonable(coords=water_dimer.coordinates)


def test_conformer_generation():

    Config.num_complex_random_rotations = 2
    Config.num_complex_sphere_points = 6
    Config.max_num_complex_conformers = 10000

    trimer._generate_conformers()
    assert len(trimer.conformers) == 6**2 * 2**2

    # all_atoms = []
    # for conf in trimer.conformers:
    #     all_atoms += conf.atoms

    # from autode.input_output import atoms_to_xyz_file
    # atoms_to_xyz_file(atoms=all_atoms, filename='tmp.xyz')


def test_conformer_generation2():

    Config.num_complex_random_rotations = 1
    Config.num_complex_sphere_points = 6
    Config.max_num_complex_conformers = 10000

    dimer._generate_conformers()
    assert len(dimer.conformers) == 6

    Config.num_complex_random_rotations = 2
    Config.max_num_complex_conformers = 10000

    dimer._generate_conformers()
    assert len(dimer.conformers) == 6 * 2


def test_complex_init():

    h2o = Molecule(
        name="water", atoms=[Atom("O"), Atom("H", x=-1), Atom("H", x=1)]
    )

    h2o_dimer = Complex(h2o, h2o, do_init_translation=False, copy=False)
    h2o.translate([1.0, 0.0, 0.0])

    # Shifting one molecule without a copy should result in both molecules
    # within the complex being translated, thus the O-O distance being 0
    assert h2o_dimer.distance(0, 3) == 0.0

    # (check the atoms have moved)
    assert np.linalg.norm(h2o_dimer.atoms[0].coord) > 0.9

    # but not if the molecules are copied
    h2o = Molecule(
        name="water", atoms=[Atom("O"), Atom("H", x=-1), Atom("H", x=1)]
    )
    h2o_dimer = Complex(h2o, h2o, do_init_translation=False, copy=True)

    h2o_dimer.translate_mol([1.0, 0.0, 0.0], mol_index=1)
    assert h2o_dimer.distance(0, 3) > 0.9

    # (original molecule should not have moved
    assert -1e-4 < np.linalg.norm(h2o.atoms[0].coord) < 1e-4


def test_complex_atom_reorder():

    hf_dimer = Complex(
        Molecule(name="HF", atoms=[Atom("H"), Atom("F", x=1.0)]),
        Molecule(name="HF", atoms=[Atom("H"), Atom("F", x=1.0)]),
    )

    with pytest.raises(Exception):
        _ = hf_dimer.atom_indexes(2)  # molecules are indexed from 0

    assert [atom.label for atom in hf_dimer.atoms] == ["H", "F", "H", "F"]

    hf_dimer.reorder_atoms(mapping={0: 1, 1: 0, 2: 2, 3: 3})
    assert [atom.label for atom in hf_dimer.atoms] == ["F", "H", "H", "F"]
    assert hf_dimer.n_molecules == 2


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
@testutils.requires_with_working_xtb_install
def test_allow_connectivity_change():

    xtb = XTB()
    xtb.path = shutil.which("xtb")
    assert xtb.is_available

    na_h2o = NCIComplex(Molecule(smiles="[Na+]"), Molecule(smiles="O"))

    # Should prune connectivity change
    try:
        na_h2o.find_lowest_energy_conformer(lmethod=xtb)
        assert na_h2o.n_conformers == 0

    # Will fail to set the lowest energy conformer
    except (NoConformers, RuntimeError):
        pass

    # but should generate more conformers allowing the Na-OH2 'bond'
    na_h2o.find_lowest_energy_conformer(allow_connectivity_changes=True)
    assert na_h2o.n_conformers > 0
