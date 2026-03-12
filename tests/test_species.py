from autode.species.species import Species
from autode.species.molecule import Molecule
from autode.wrappers.ORCA import orca
from autode.wrappers.XTB import xtb
from autode.calculations import Calculation
from autode.conformers import Conformers
from autode.atoms import Atom
from autode.solvent.solvents import Solvent
from autode.solvent.solvents import get_solvent
from autode.geom import calc_rmsd
from autode.values import Gradient, EnthalpyCont, PotentialEnergy
from autode.units import ha_per_ang
from autode.exceptions import NoAtomsInMolecule, CalculationException
from autode.utils import work_in_tmp_dir
from scipy.spatial import distance_matrix
from copy import deepcopy
from . import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))

h1 = Atom("H")
h2 = Atom("H", z=1.0)

mol = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)


def test_species_class():
    blank_mol = Species(name="tmp", atoms=None, charge=0, mult=1)

    assert blank_mol.n_atoms == 0
    assert blank_mol.n_conformers == 0
    assert blank_mol.radius == 0

    assert str(blank_mol) != ""  # Should have some string representations
    assert repr(blank_mol) != ""

    assert blank_mol.has_reasonable_coordinates  # No coordinates are good

    assert hasattr(mol, "print_xyz_file")
    assert hasattr(mol, "translate")
    assert hasattr(mol, "rotate")
    assert hasattr(mol, "coordinates")
    assert str(mol) != ""

    assert mol.charge == 0
    assert mol.mult == 1
    assert mol.name == "H2"

    for attr in (
        "gradient",
        "hessian",
        "free_energy",
        "enthalpy",
        "g_cont",
        "h_cont",
        "frequencies",
        "vib_frequencies",
        "imaginary_frequencies",
    ):
        assert getattr(mol, attr) is None

    assert mol.normal_mode(mode_number=1) is None

    assert not mol.is_explicitly_solvated

    # A not very sensible water geometry!
    water = Species(
        name="H2O",
        charge=0,
        mult=1,
        atoms=[Atom("O"), Atom("H", z=-1), Atom("H", z=1)],
    )

    assert water.formula == "H2O" or water.formula == "OH2"

    # Species without a molecular graph (no atoms) cannot define a bond matrix
    with pytest.raises(Exception):
        _ = Molecule().bond_matrix

    # very approximate molecular radius
    assert 0.5 < water.radius < 2.5

    # Base class for molecules and TSs and complexes shouldn't have a
    # implemented conformer method – needs to do different things based on the
    # type of species to find conformers
    with pytest.raises(NotImplementedError):
        water.find_lowest_energy_conformer(lmethod=xtb)

    # Cannot optimise a molecule without a method or a calculation
    with pytest.raises(ValueError):
        water.optimise()


def test_species_energies_reset():
    tmp_species = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)
    tmp_species.energy = 1.0

    assert len(tmp_species.energies) == 1

    # At the same geometry other energies are retained, even if energy=None(?)
    tmp_species.energy = None
    assert len(tmp_species.energies) == 1

    # Translating the molecule should leave the energy unchanged
    tmp_species.atoms = [Atom("H", z=1.0), Atom("H", z=2.0)]
    assert tmp_species.energy == 1.0

    # and also rotating it
    tmp_species.atoms = [Atom("H"), Atom("H", x=1.0)]
    assert tmp_species.energy == 1.0

    # but adjusting the distance should reset the energy
    tmp_species.atoms = [Atom("H"), Atom("H", z=1.1)]
    assert tmp_species.energy is None

    # changing the number of atoms should reset the energy
    tmp_species.energy = 1.0
    tmp_species.atoms = [Atom("H")]
    assert tmp_species.energy is None

    # likewise changing the atom number
    tmp_species.atoms = [Atom("H"), Atom("H", x=1.0)]
    tmp_species.energy = 1.0
    tmp_species.atoms = [Atom("H"), Atom("F", x=1.0)]

    assert tmp_species.energy is None


def test_connectivity():
    _h2 = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)
    _h2.reset_graph()

    # Must have the same connectivity as itself
    assert _h2.has_same_connectivity_as(_h2)

    # Graphs are lazy loaded so if undefined one is built
    _h2_no_set = _h2.copy()
    _h2_no_set.graph = None
    _h2.has_same_connectivity_as(_h2_no_set)

    # Or something without a graph attribute is passed
    with pytest.raises(ValueError):
        _h2.has_same_connectivity_as("a")

    # Different number of atoms have different connectivity
    assert not _h2.has_same_connectivity_as(Molecule(atoms=None))

    # No atom molecule have the same connectivity
    assert Molecule(atoms=None).has_same_connectivity_as(Molecule(atoms=None))


def test_species_xyz_file():
    mol.print_xyz_file()
    assert os.path.exists("H2.xyz")
    xyz_file_lines = open("H2.xyz", "r").readlines()

    # First item in the xyz file needs to be the number of atoms
    assert int(xyz_file_lines[0].split()[0]) == 2

    # Third line needs to be in the format H, x, y, z
    assert len(xyz_file_lines[2].split()) == 4

    os.remove("H2.xyz")

    mol_copy = mol.copy()
    mol_copy.atoms = None

    with pytest.raises(NoAtomsInMolecule):
        mol_copy.print_xyz_file()


@work_in_tmp_dir()
def test_species_mol_file():
    h2_mol = Molecule(
        atoms=[
            Atom("H", -1.6030, 0.6090, 0.0),
            Atom("H", -0.8950, 0.6182, 0.0),
        ]
    )
    with pytest.raises(AssertionError):
        h2_mol.print_mol_file("H2.xyz")

    h2_mol.print_mol_file("H2.mol")
    assert os.path.isfile("H2.mol")

    with open("H2.mol") as fh:
        molfile_lines = fh.readlines()

    # first line is the name of molecule
    assert molfile_lines[0].strip() == "H2"
    # second line the software that was used to print
    assert molfile_lines[1].strip() == "autodE"
    # third line is empty
    assert molfile_lines[2].strip() == ""

    # third line has number of atoms, number of bonds, preset fields
    assert molfile_lines[3] == "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    # output tested with Avogadro
    assert molfile_lines[4:] == [
        "  -1.60300   0.60900   0.00000 H     0  0"
        "  0  0  0  0  0  0  0  0  0  0\n",
        "  -0.89500   0.61820   0.00000 H     0  0"
        "  0  0  0  0  0  0  0  0  0  0\n",
        "  1  2  1  0  0  0  0\n",
        "M  END\n",
    ]

    h2_mol.print_mol_file()
    assert os.path.isfile(f"{h2_mol.name}.mol")


def test_species_translate():
    m = Species(
        name="H2", atoms=[Atom("H"), Atom("H", z=1.0)], charge=0, mult=1
    )
    m.translate(vec=np.array([0.0, 0.0, -1.0]))

    expected = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

    assert np.allclose(m.atoms[0].coord, expected[0, :])
    assert np.allclose(m.atoms[1].coord, expected[1, :])
    assert np.allclose(m.coordinates, expected)

    # Centering should move the middle of the molecule to the origin
    m.centre()
    assert np.allclose(
        np.average(m.coordinates, axis=0), np.zeros(3), atol=1e-4
    )


def test_species_rotate():
    m = Species(
        name="H2", atoms=[Atom("H"), Atom("H", z=1.0)], charge=0, mult=1
    )
    # Rotation about the y axis 180 degrees (π radians)
    m.rotate(axis=np.array([1.0, 0.0, 0.0]), theta=np.pi)

    assert np.linalg.norm(m.atoms[0].coord - np.array([0.0, 0.0, 0.0])) < 1e-9
    assert np.linalg.norm(m.atoms[1].coord - np.array([0.0, 0.0, -1.0])) < 1e-9


def test_get_coordinates():
    coords = mol.coordinates
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (2, 3)


def test_set_atoms():
    mol_copy = deepcopy(mol)

    mol_copy.atoms = [h1]
    assert mol_copy.n_atoms == 1
    assert len(mol_copy.atoms) == 1


def test_set_coords():
    mol_copy = deepcopy(mol)

    new_coords = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    mol_copy.coordinates = new_coords

    assert (
        np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, 1.0]))
        < 1e-9
    )
    assert (
        np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, 0.0]))
        < 1e-9
    )


def test_set_gradients():
    test_mol = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)

    # Gradient must be a Nx3 array for N atoms
    with pytest.raises(ValueError):
        test_mol.gradient = 5

    with pytest.raises(ValueError):
        test_mol.gradient = np.zeros(shape=(test_mol.n_atoms, 2))

    # but can set them with a Gradients array
    test_mol.gradient = Gradient(
        np.zeros(shape=(test_mol.n_atoms, 3)), units="Ha Å^-1"
    )
    assert test_mol.gradient.units == ha_per_ang

    # setting from a numpy array defaults to Ha/Å units
    test_mol.gradient = np.zeros(shape=(2, 3))
    assert test_mol.gradient.units == ha_per_ang

    # will reshape numpy array if possible
    arr = np.random.rand(6)
    test_mol.gradient = arr
    assert np.allclose(test_mol.gradient, arr.reshape(-1, 3))


def test_species_solvent():
    assert mol.solvent is None

    solvated_mol = Species(
        name="H2", atoms=[h1, h2], charge=0, mult=1, solvent_name="water"
    )
    assert isinstance(solvated_mol.solvent, Solvent)

    solvated_mol.solvent = None
    assert solvated_mol.solvent is None

    solvated_mol.solvent = "water"
    assert isinstance(solvated_mol.solvent, Solvent)


def test_reorder():
    hf = Species(
        name="HF", charge=0, mult=1, atoms=[Atom("H"), Atom("F", x=1)]
    )

    assert hf.atoms[0].label == "H" and hf.atoms[1].label == "F"

    # A simple reorder should swap the atoms
    hf.reorder_atoms(mapping={0: 1, 1: 0})
    assert hf.atoms[0].label == "F" and hf.atoms[1].label == "H"

    # Cannot reorder if the atoms if the mapping isn't 1-1
    with pytest.raises(ValueError):
        hf.reorder_atoms(mapping={0: 1, 1: 1})


@testutils.work_in_zipped_dir(os.path.join(here, "data", "species.zip"))
def test_species_single_point():
    mol.single_point(method=orca)
    assert mol.energy == -1.138965730007

    failed_sp_mol = Species(name="H2_failed", atoms=[h1, h2], charge=0, mult=1)

    with pytest.raises(CalculationException):
        failed_sp_mol.single_point(method=orca)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "species.zip"))
def test_species_optimise():
    orca.path = here
    assert orca.is_available

    dihydrogen = Species(
        name="H2", atoms=[Atom("H"), Atom("H", x=1)], charge=0, mult=1
    )

    dihydrogen.optimise(method=orca)
    assert dihydrogen.atoms is not None

    # Resetting the graph after the optimisation should still have a single
    # edge as the bond between H atoms
    dihydrogen.optimise(method=orca, reset_graph=True)
    assert len(dihydrogen.graph.edges) == 1


@testutils.work_in_zipped_dir(os.path.join(here, "data", "species.zip"))
def test_find_lowest_energy_conformer():
    # Spoof XTB availability
    xtb.path = here

    propane = Molecule(name="propane", smiles="CCC")

    propane.find_lowest_energy_conformer(lmethod=xtb)
    assert len(propane.conformers) > 0

    # Finding low energy conformers should set the energy of propane
    assert propane.energy is not None
    assert propane.atoms is not None


def test_species_copy():
    species = Species(name="h", charge=0, mult=2, atoms=[Atom("H")])

    species_copy = species.copy()
    species_copy.charge = 1

    assert species.charge != species_copy.charge

    species_copy.mult = 3
    assert species.mult != species_copy.mult

    atom = species_copy.atoms[0]
    atom.translate(vec=np.array([1.0, 1.0, 1.0]))
    assert np.linalg.norm(species.atoms[0].coord - atom.coord) > 1


def test_species_formula():
    assert mol.formula == "H2"

    mol_no_atoms = Molecule()
    assert mol_no_atoms.formula == ""


def test_generate_conformers():
    with pytest.raises(NotImplementedError):
        mol._generate_conformers()


def test_set_lowest_energy_conformer():
    hb = Atom("H", z=0.7)
    hydrogen = Species(name="H2", atoms=[h1, hb], charge=0, mult=1)

    hydrogen_wo_e = Species(name="H2", atoms=[h1, hb], charge=0, mult=1)

    hydrogen_with_e = Species(name="H2", atoms=[h1, hb], charge=0, mult=1)
    hydrogen_with_e.energy = -1

    hydrogen.conformers = [hydrogen_wo_e, hydrogen_with_e]
    hydrogen._set_lowest_energy_conformer()

    # Conformers without energy should be skipped
    assert hydrogen.energy == -1

    # Conformers with a different molecular graph should be skipped
    h_atom = Species(name="H", atoms=[Atom("H")], charge=0, mult=1)
    h_atom.energy = -2
    hydrogen.conformers = [hydrogen_with_e, h_atom]

    assert hydrogen.energy == -1


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_thermal_cont_without_hess_run():
    calc = Calculation(
        name="test", molecule=mol, method=orca, keywords=orca.keywords.hess
    )
    mol.energy = -1

    # Some blank output that exists
    calc.output.filename = "test.out"
    with open("test.out", "w") as out:
        print("test", file=out)

    assert calc.output.exists

    # Calculating the free energy contribution without a correct Hessian

    with pytest.raises(Exception):
        mol.calc_g_cont(calc=calc)

    # and similarly with the enthalpic contribution
    with pytest.raises(Exception):
        mol.calc_h_cont(calc=calc)


def test_is_linear():
    h_atom = Species(name="h", atoms=[Atom("H")], charge=0, mult=1)
    assert not h_atom.is_linear()

    dihydrogen = Species(
        name="h2", atoms=[Atom("H"), Atom("H", x=1)], charge=0, mult=1
    )
    assert dihydrogen.is_linear()

    water = Species(
        name="water",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", x=-1.52, y=2.72),
            Atom("H", x=-0.54, y=2.72),
            Atom("H", x=-1.82, y=2.82, z=-0.92),
        ],
    )
    assert not water.is_linear()
    assert water.is_planar()

    lin_water = Species(
        name="linear_water",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", x=-1.52, y=2.72),
            Atom("H", x=-1.21, y=2.51, z=1.03),
            Atom("H", x=-1.82, y=2.82, z=-0.92),
        ],
    )
    assert lin_water.is_linear(tol=0.01)

    close_lin_water = Species(
        name="linear_water",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", x=-1.52, y=2.72),
            Atom("H", x=-0.90, y=2.36, z=0.89),
            Atom("H", x=-1.82, y=2.82, z=-0.92),
        ],
    )
    assert not close_lin_water.is_linear()

    acetylene = Molecule(smiles="C#C")
    assert acetylene.is_linear(tol=0.01)


def test_unique_conformer_set():
    test_mol = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)
    test_mol.energy = -1.0

    # With the same molecule the conformer list will be pruned to 1
    test_mol.conformers = [test_mol.copy(), test_mol.copy()]
    test_mol.conformers.prune_on_energy()
    assert len(test_mol.conformers) == 1

    test_mol.conformers = None
    assert type(test_mol.conformers) is Conformers
    assert test_mol.n_conformers == 0


def test_unique_conformer_set_energy():
    # or where one conformer has a very different energy
    test_mol = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)
    test_mol.energy = -1.0

    test_mol_high_e = test_mol.copy()
    test_mol_high_e.energy = 10.0
    test_mol.conformers = [test_mol_high_e, test_mol.copy(), test_mol.copy()]
    test_mol.conformers.prune_on_energy(n_sigma=1)

    assert len(test_mol.conformers) == 1
    assert test_mol.conformers[0].energy == -1.0


@testutils.work_in_zipped_dir(os.path.join(here, "data", "species.zip"))
def test_hessian_calculation():
    h2o = Species(
        name="H2O",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", -0.0011, 0.3631, -0.0),
            Atom("H", -0.8250, -0.1819, -0.0),
            Atom("H", 0.8261, -0.1812, 0.0),
        ],
    )

    # Spoof ORCA install
    orca.path = here
    assert orca.is_available

    h2o._run_hess_calculation(method=orca)
    assert h2o.hessian is not None
    assert h2o.frequencies is not None


def test_numerical_hessian_invalid_delta():
    with pytest.raises(ValueError):
        mol.calc_hessian(method=orca, coordinate_shift="a", numerical=True)


def test_enthalpy_doc_example():
    _h2 = Molecule(smiles="[H][H]")
    _h2.energies.append(EnthalpyCont(0.0133, units="Ha"))
    _h2.energy = PotentialEnergy(
        -1.16397, units="Ha", method=orca, keywords=orca.keywords.opt
    )
    assert np.isclose(_h2.enthalpy, -1.15067, atol=1e-4)

    _h2.energy = PotentialEnergy(
        -1.16827, units="Ha", method=orca, keywords=orca.keywords.sp
    )

    assert np.isclose(_h2.enthalpy, -1.15497, atol=1e-4)


def test_species_rotation_preserves_internals():
    methane = Molecule(smiles="C")
    init_coords = methane.coordinates

    axes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],  # Need-not be normalised
        [2.0, 0.1, 0.3],
    ]
    thetas = [0.1, 1.0, 3.14159, 5.6]

    for axis in axes:
        for theta in thetas:
            methane.rotate(axis=axis, theta=theta)
            # Rotation should preserve the relative positions i.e. a small RMSD
            assert calc_rmsd(methane.coordinates, init_coords) < 0.01

            # Shift back to original coordinates
            methane.coordinates = init_coords


def test_species_rotation_is_same_as_atom():
    water = Molecule(smiles="O")
    water_atoms = water.atoms.copy()

    axis, angle = [0.2, 0.7, -0.3], 2.41
    water.rotate(axis=axis, theta=angle)
    for atom in water_atoms:
        atom.rotate(axis=axis, theta=angle)

    assert np.linalg.norm((water.coordinates - water_atoms.coordinates)) < 0.01


@testutils.work_in_zipped_dir(os.path.join(here, "data", "species.zip"))
def test_keywords_opt_sp_thermo():
    h2o = Molecule(smiles="O", name="water_tmp")
    orca.path = here
    assert orca.is_available

    # Check that the calculations work with keywords specified as either a
    # regular list or as a single string
    for kwds in (["Opt", "def2-SVP", "PBE"], "Opt def2-SVP PBE"):
        h2o.energies.clear()
        h2o.optimise(method=orca, keywords=kwds)
        assert h2o.energy is not None

    for kwds in (["SP", "def2-SVP", "PBE"], "SP def2-SVP PBE"):
        h2o.energies.clear()
        h2o.single_point(method=orca, keywords=kwds)
        assert h2o.energy is not None

    for kwds in (["Freq", "def2-SVP", "PBE"], "Freq def2-SVP PBE"):
        h2o.energies.clear()
        h2o.hessian = None
        h2o.calc_thermo(method=orca, keywords=kwds)
        assert h2o.energy is not None


def test_flat_species_has_reasonable_coordinates():
    c2h4 = Molecule(
        atoms=[
            Atom("C", -4.99490, 1.95320, 0.00000),
            Atom("C", -4.74212, 0.64644, 0.00000),
            Atom("H", -4.17835, 2.66796, 0.00000),
            Atom("H", -6.01909, 2.31189, 0.00000),
            Atom("H", -3.71793, 0.28776, -0.00000),
            Atom("H", -5.55867, -0.06831, 0.00000),
        ]
    )

    assert c2h4.has_reasonable_coordinates

    rh_h4 = Molecule(
        atoms=[
            Atom("Rh", -0.19569, -2.70701, -0.00000),
            Atom("H", -0.62458, -1.07785, 0.00000),
            Atom("H", -1.82557, -3.13312, -0.00000),
            Atom("H", 1.43448, -2.28115, 0.00000),
            Atom("H", 0.23390, -4.33620, -0.00000),
        ]
    )

    assert rh_h4.graph.number_of_edges() == 4

    # [Rh(H)4] does have a reasonable flat structure, as a square planar
    # geometry is possible
    assert rh_h4.has_reasonable_coordinates


def test_species_does_not_have_reasonable_coordinates():
    ch4_flat = Molecule(
        atoms=[
            Atom("C", 0.0, 0.0, 0.0),
            Atom("H", 0.0, -1.0, 0.0),
            Atom("H", 0.0, 1.0, 0.0),
            Atom("H", 0.0, 0.0, -1.0),
            Atom("H", 0.0, 0.0, 1.0),
        ]
    )

    # CH4 should not be flat
    assert not ch4_flat.has_reasonable_coordinates

    x = ch4_flat.coordinates
    assert np.min(distance_matrix(x, x) + np.eye(5)) > 0.7


@testutils.requires_working_xtb_install
def test_calc_thermo_not_run_calculation():
    m = Molecule(smiles="O")
    calc = Calculation(
        name="water", molecule=m, method=xtb, keywords=xtb.keywords.hess
    )
    # run() has not been called
    with pytest.raises(Exception):
        m.calc_thermo(calc=calc)


@pytest.mark.parametrize("mult", [1, 3, 5])
def test_argon_has_valid_spin_state(mult: int, charge: int = 0):
    assert Molecule(
        atoms=[Atom("Ar")], mult=mult, charge=charge
    ).has_valid_spin_state


@pytest.mark.parametrize("mult", [1, 3, 4])
def test_hydrogen_has_invalid_spin_state(mult: int, charge: int = 0):
    assert not Molecule(
        atoms=[Atom("H")], mult=mult, charge=charge
    ).has_valid_spin_state


def test_has_valid_spin_state_docstring():
    assert not Molecule(
        atoms=[Atom("H")], charge=0, mult=1
    ).has_valid_spin_state
    assert Molecule(atoms=[Atom("H")], charge=-1, mult=1).has_valid_spin_state


@pytest.mark.parametrize("invalid_mult", [0, -1, "a", (0, 2)])
def test_cannot_set_multiplicity_to_invalid_value(invalid_mult):
    m = Species(name="H2", atoms=[h1, h2], charge=0, mult=1)
    with pytest.raises(Exception):
        m.mult = invalid_mult


@work_in_tmp_dir()
def test_cant_init_with_both_xyz_and_smiles():
    filename = "tmp.xyz"
    tmp_mol = Molecule(smiles="O")
    tmp_mol.print_xyz_file(filename=filename)

    with pytest.raises(AssertionError):
        _ = Molecule("tmp.xyz", smiles="[H]S[H]")


@work_in_tmp_dir()
def test_species_load_from_xyz_file_retains_spin_mult_and_solvent():
    filename = "tmp.xyz"
    tmp_mol = Molecule(smiles="[O+]", mult=2, charge=1, solvent_name="water")
    tmp_mol.print_xyz_file(filename=filename)

    loaded_mol = Molecule(filename)
    assert loaded_mol.mult == 2
    assert loaded_mol.charge == 1
    assert loaded_mol.solvent == get_solvent(
        solvent_name="water", kind="implicit"
    )
    assert loaded_mol.solvent.is_implicit
