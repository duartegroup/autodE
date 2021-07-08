from autode.species.species import Species
from autode.species.molecule import Molecule
from autode.wrappers.ORCA import orca
from autode.wrappers.XTB import xtb
from autode.calculation import Calculation
from autode.atoms import Atom
from autode.solvent.solvents import Solvent
from autode.values import Gradient
from autode.units import ha_per_ang
from autode.exceptions import NoAtomsInMolecule, CalculationException
from autode.utils import work_in_tmp_dir
from copy import deepcopy
from . import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))

h1 = Atom('H')
h2 = Atom('H', z=1.0)

mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1)


def test_species_class():

    blank_mol = Species(name='tmp', atoms=None, charge=0, mult=1)

    assert blank_mol.n_atoms == 0
    assert blank_mol.radius == 0

    assert str(blank_mol) != ''   # Should have some string representations
    assert repr(blank_mol) != ''

    assert hasattr(mol, 'print_xyz_file')
    assert hasattr(mol, 'translate')
    assert hasattr(mol, 'rotate')
    assert hasattr(mol, 'coordinates')
    assert str(mol) != ''

    assert mol.charge == 0
    assert mol.mult == 1
    assert mol.name == 'H2'

    for attr in ('gradient', 'hessian', 'free_energy', 'enthalpy', 'g_cont',
                 'h_cont', 'frequencies', 'vib_frequencies',
                 'imaginary_frequencies'):
        assert getattr(mol, attr) is None

    assert mol.normal_mode(mode_number=1) is None

    assert not mol.is_explicitly_solvated

    # A not very sensible water geometry!
    water = Species(name='H2O', charge=0, mult=1,
                    atoms=[Atom('O'), Atom('H', z=-1), Atom('H', z=1)])

    assert water.formula == 'H2O' or water.formula == 'OH2'

    # Species without a molecular graph cannot define a bond matrix
    with pytest.raises(ValueError):
        _ = water.bond_matrix

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

    tmp_species = Species(name='H2', atoms=[h1, h2], charge=0, mult=1)
    tmp_species.energy = 1.0

    assert len(tmp_species.energies) == 1

    # At the same geometry other energies are retained, even if energy=None(?)
    tmp_species.energy = None
    assert len(tmp_species.energies) == 1

    # Translating the molecule should leave the energy unchanged
    tmp_species.atoms = [Atom('H', z=1.0), Atom('H', z=2.0)]
    assert tmp_species.energy == 1.0

    # and also rotating it
    tmp_species.atoms = [Atom('H'), Atom('H', x=1.0)]
    assert tmp_species.energy == 1.0

    # but adjusting the distance should reset the energy
    tmp_species.atoms = [Atom('H'), Atom('H', z=1.1)]
    assert tmp_species.energy is None

    # changing the number of atoms should reset the energy
    tmp_species.energy = 1.0
    tmp_species.atoms = [Atom('H')]
    assert tmp_species.energy is None

    # likewise changing the atom number
    tmp_species.atoms = [Atom('H'), Atom('H', x=1.0)]
    tmp_species.energy = 1.0
    tmp_species.atoms = [Atom('H'), Atom('F', x=1.0)]

    assert tmp_species.energy is None


def test_species_xyz_file():

    mol.print_xyz_file()
    assert os.path.exists('H2.xyz')
    xyz_file_lines = open('H2.xyz', 'r').readlines()

    # First item in the xyz file needs to be the number of atoms
    assert int(xyz_file_lines[0].split()[0]) == 2

    # Third line needs to be in the format H, x, y, z
    assert len(xyz_file_lines[2].split()) == 4

    os.remove('H2.xyz')

    mol_copy = mol.copy()
    mol_copy.atoms = None

    with pytest.raises(NoAtomsInMolecule):
        mol_copy.print_xyz_file()


def test_species_translate():
    mol_copy = deepcopy(mol)
    mol_copy.translate(vec=np.array([0.0, 0.0, -1.0]))

    assert np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, -1.0])) < 1E-9
    assert np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, 0.0])) < 1E-9

    # Centering should move the middle of the molecule to the origin
    mol_copy.centre()
    assert np.allclose(np.average(mol_copy.coordinates, axis=0),
                       np.zeros(3),
                       atol=1E-4)


def test_species_rotate():
    mol_copy = deepcopy(mol)
    # Rotation about the y axis 180 degrees (π radians)
    mol_copy.rotate(axis=np.array([1.0, 0.0, 0.0]), theta=np.pi)

    assert np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, 0.0])) < 1E-9
    assert np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, -1.0])) < 1E-9


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

    new_coords = np.array([[0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0]])

    mol_copy.coordinates = new_coords

    assert np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, 1.0])) < 1E-9
    assert np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, 0.0])) < 1E-9


def test_set_gradients():

    test_mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1)

    # Gradient must be a Nx3 array for N atoms
    with pytest.raises(ValueError):
        test_mol.gradient = 5

    with pytest.raises(ValueError):
        test_mol.gradient = np.zeros(shape=(test_mol.n_atoms, 2))

    # but can set them with a Gradients array
    test_mol.gradient = Gradient(np.zeros(shape=(test_mol.n_atoms, 3)),
                                 units='Ha Å^-1')
    assert test_mol.gradient.units == ha_per_ang

    # setting from a numpy array defaults to Ha/Å units
    test_mol.gradient = np.zeros(shape=(2, 3))
    assert test_mol.gradient.units == ha_per_ang


def test_species_solvent():

    assert mol.solvent is None

    solvated_mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1, solvent_name='water')
    assert type(solvated_mol.solvent) == Solvent


def test_reorder():
    hf = Species(name='HF', charge=0, mult=1,
                 atoms=[Atom('H'), Atom('F', x=1)])

    assert hf.atoms[0].label == 'H' and hf.atoms[1].label == 'F'

    # A simple reorder should swap the atoms
    hf.reorder_atoms(mapping={0: 1, 1: 0})
    assert hf.atoms[0].label == 'F' and hf.atoms[1].label == 'H'

    # Cannot reorder if the atoms if the mapping isn't 1-1
    with pytest.raises(ValueError):
        hf.reorder_atoms(mapping={0: 1, 1: 1})


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'species.zip'))
def test_species_single_point():

    mol.single_point(method=orca)
    assert mol.energy == -1.138965730007


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'species.zip'))
def test_species_optimise():

    orca.path = here
    assert orca.available

    dihydrogen = Species(name='H2', atoms=[Atom('H'), Atom('H', x=1)],
                         charge=0, mult=1)

    dihydrogen.optimise(method=orca)
    assert dihydrogen.atoms is not None

    # Resetting the graph after the optimisation should still have a single
    # edge as the bond between H atoms
    dihydrogen.optimise(method=orca, reset_graph=True)
    assert len(dihydrogen.graph.edges) == 1


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'species.zip'))
def test_find_lowest_energy_conformer():

    # Spoof XTB availability
    xtb.path = here

    propane = Molecule(name='propane', smiles='CCC')

    propane.find_lowest_energy_conformer(lmethod=xtb)
    assert len(propane.conformers) > 0

    # Finding low energy conformers should set the energy of propane
    assert propane.energy is not None
    assert propane.atoms is not None


def test_species_copy():

    species = Species(name='h', charge=0, mult=2, atoms=[Atom('H')])

    species_copy = species.copy()
    species_copy.charge = 1

    assert species.charge != species_copy.charge

    species_copy.mult = 3
    assert species.mult != species_copy.mult

    atom = species_copy.atoms[0]
    atom.translate(vec=np.array([1.0, 1.0, 1.0]))
    assert np.linalg.norm(species.atoms[0].coord - atom.coord) > 1


def test_species_formula():

    assert mol.formula == 'H2'
    
    mol_no_atoms = Molecule()
    assert mol_no_atoms.formula == ""


def test_generate_conformers():

    with pytest.raises(NotImplementedError):
        mol._generate_conformers()


def test_set_lowest_energy_conformer():

    from autode.mol_graphs import make_graph

    hb = Atom('H', z=0.7)
    hydrogen = Species(name='H2', atoms=[h1, hb], charge=0, mult=1)
    make_graph(hydrogen)

    hydrogen_wo_e = Species(name='H2', atoms=[h1, hb], charge=0, mult=1)

    hydrogen_with_e = Species(name='H2', atoms=[h1, hb], charge=0, mult=1)
    hydrogen_with_e.energy = -1

    hydrogen.conformers = [hydrogen_wo_e, hydrogen_with_e]
    hydrogen._set_lowest_energy_conformer()

    # Conformers without energy should be skipped
    assert hydrogen.energy == -1

    # Conformers with a different molecular graph should be skipped
    h_atom = Species(name='H', atoms=[Atom('H')], charge=0, mult=1)
    h_atom.energy = -2
    hydrogen.conformers = [hydrogen_with_e, h_atom]

    assert hydrogen.energy == -1


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_thermal_cont_without_hess_run():

    calc = Calculation(name='test',
                       molecule=mol,
                       method=orca,
                       keywords=orca.keywords.hess)
    mol.energy = -1

    # Some blank output that exists
    calc.output.filename = 'test.out'
    with open('test.out', 'w') as out:
        print('test', file=out)

    assert calc.output.exists

    # Calculating the free energy contribution without a correct Hessian

    with pytest.raises(CalculationException):
        mol.calc_g_cont(calc=calc)

    # and similarly with the enthalpic contribution
    with pytest.raises(CalculationException):
        mol.calc_h_cont(calc=calc)


def test_is_linear():

    h_atom = Species(name='h', atoms=[Atom('H')], charge=0, mult=1)
    assert not h_atom.is_linear()

    dihydrogen = Species(name='h2', atoms=[Atom('H'), Atom('H', x=1)],
                         charge=0, mult=1)
    assert dihydrogen.is_linear()

    water = Species(name='water', charge=0, mult=1,
                    atoms=[Atom('O', x=-1.52, y=2.72),
                           Atom('H', x=-0.54, y=2.72),
                           Atom('H', x=-1.82, y=2.82, z=-0.92)])
    assert not water.is_linear()

    lin_water = Species(name='linear_water', charge=0, mult=1,
                        atoms=[Atom('O', x=-1.52, y=2.72),
                               Atom('H', x=-1.21, y=2.51, z=1.03),
                               Atom('H', x=-1.82, y=2.82, z=-0.92)])
    assert lin_water.is_linear(tol=0.01)

    close_lin_water = Species(name='linear_water', charge=0, mult=1,
                              atoms=[Atom('O', x=-1.52, y=2.72),
                                     Atom('H', x=-0.90, y=2.36, z=0.89),
                                     Atom('H', x=-1.82, y=2.82, z=-0.92)])
    assert not close_lin_water.is_linear()

    acetylene = Molecule(smiles='C#C')
    assert acetylene.is_linear(tol=0.01)


def test_unique_conformer_set():

    test_mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1)
    test_mol.energy = -1.0

    # With the same molecule the conformer list will be pruned to 1
    conformers = [test_mol.copy(), test_mol.copy()]
    test_mol._set_unique_conformers_rmsd(conformers)
    assert len(test_mol.conformers) == 1


def test_unique_conformer_set_energy():

    # or where one conformer has a very different energy
    test_mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1)
    test_mol.energy = -1.0

    test_mol_high_e = test_mol.copy()
    test_mol_high_e.energy = 10.0
    conformers = [test_mol_high_e, test_mol.copy(), test_mol.copy()]
    test_mol._set_unique_conformers_rmsd(conformers, n_sigma=1)

    assert len(test_mol.conformers) == 1
    assert test_mol.conformers[0].energy == -1.0


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'species.zip'))
def test_hessian_calculation():

    h2o = Species(name='H2O', charge=0, mult=1,
                  atoms=[Atom('O', -0.0011,  0.3631, -0.0),
                         Atom('H', -0.8250, -0.1819, -0.0),
                         Atom('H',  0.8261, -0.1812,  0.0)])

    # Spoof ORCA install
    orca.path = here
    assert orca.available

    h2o._run_hess_calculation(method=orca)
    assert h2o.hessian is not None
    assert h2o.frequencies is not None
