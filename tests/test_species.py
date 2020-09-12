from autode.species.species import Species
from autode.species.molecule import Molecule
from autode.wrappers.ORCA import orca
from autode.wrappers.XTB import xtb
from autode.atoms import Atom
from autode.solvent.solvents import Solvent
from autode.exceptions import NoAtomsInMolecule
from copy import deepcopy
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))

h1 = Atom('H')
h2 = Atom('H', z=1.0)

mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1)


def test_species_class():

    assert hasattr(mol, 'print_xyz_file')
    assert hasattr(mol, 'translate')
    assert hasattr(mol, 'rotate')
    assert hasattr(mol, 'get_coordinates')
    assert hasattr(mol, 'set_atoms')
    assert hasattr(mol, 'set_coordinates')

    assert mol.charge == 0
    assert mol.mult == 1
    assert mol.name == 'H2'

    assert not mol.is_explicitly_solvated()


def test_species_xyz_file():

    mol.print_xyz_file()
    assert os.path.exists('H2.xyz')
    xyz_file_lines = open('H2.xyz', 'r').readlines()
    assert int(xyz_file_lines[0].split()[0]) == 2  # First item in the xyz file needs to be the number of atoms
    assert len(xyz_file_lines[2].split()) == 4  # Third line needs to be in the format H, x, y, z

    os.remove('H2.xyz')

    mol_copy = deepcopy(mol)
    mol_copy.atoms = mol_copy.set_atoms(atoms=None)

    with pytest.raises(NoAtomsInMolecule):
        mol_copy.print_xyz_file()


def test_species_translate():
    mol_copy = deepcopy(mol)
    mol_copy.translate(vec=np.array([0.0, 0.0, -1.0]))

    assert np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, -1.0])) < 1E-9
    assert np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, 0.0])) < 1E-9


def test_species_rotate():
    mol_copy = deepcopy(mol)
    mol_copy.rotate(axis=np.array([1.0, 0.0, 0.0]), theta=np.pi)     # Rotation about the y axis 180 degrees (π radians)

    assert np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, 0.0])) < 1E-9
    assert np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, -1.0])) < 1E-9


def test_get_coordinates():

    coords = mol.get_coordinates()
    assert type(coords) == np.ndarray
    assert coords.shape == (2, 3)


def test_set_atoms():
    mol_copy = deepcopy(mol)

    mol_copy.set_atoms(atoms=[h1])
    assert mol_copy.n_atoms == 1
    assert len(mol_copy.atoms) == 1


def test_set_coords():
    mol_copy = deepcopy(mol)

    new_coords = np.array([[0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0]])

    mol_copy.set_coordinates(coords=new_coords)

    assert np.linalg.norm(mol_copy.atoms[0].coord - np.array([0.0, 0.0, 1.0])) < 1E-9
    assert np.linalg.norm(mol_copy.atoms[1].coord - np.array([0.0, 0.0, 0.0])) < 1E-9


def test_species_solvent():

    assert mol.solvent is None

    solvated_mol = Species(name='H2', atoms=[h1, h2], charge=0, mult=1, solvent_name='water')
    assert type(solvated_mol.solvent) == Solvent


def test_species_single_point():

    os.chdir(os.path.join(here, 'data'))

    orca.available = True
    mol.single_point(method=orca)
    assert mol.energy == -1.138965730007

    os.remove('H2_sp_orca.inp')
    os.chdir(here)


def test_find_lowest_energy_conformer():

    os.chdir(os.path.join(here, 'data'))

    # Spoof XTB availability
    xtb.path = here
    xtb.available = True

    propane = Molecule(name='propane', smiles='CCC')

    propane.find_lowest_energy_conformer(lmethod=xtb)
    assert len(propane.conformers) > 0

    # Finding low energy conformers should set the energy of propane
    assert propane.energy is not None
    assert propane.atoms is not None

    os.chdir(here)


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

    assert mol.formula() == 'H2'
    
    mol_no_atoms = Molecule()
    assert mol_no_atoms.formula() is None


def test_generate_conformers():

    with pytest.raises(NotImplementedError):
        mol._generate_conformers()
