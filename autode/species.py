import numpy as np
from autode.log import logger
from autode.solvent.solvents import get_solvent
from autode.calculation import Calculation
from autode.config import Config
from autode.utils import requires_atoms


class Species:

    @requires_atoms()
    def translate(self, vec):
        """Translate the molecule by vector (np.ndarray, length 3)"""
        for atom in self.atoms:
            atom.translate(vec)
        return None

    @requires_atoms()
    def rotate(self, axis, theta, origin=None):
        """Rotate the molecule by around an axis (np.ndarray, length 3) an theta radians"""
        for atom in self.atoms:

            # Shift so that the origin is at (0, 0, 0), apply the rotation, and shift back
            if origin is not None:
                atom.translate(vec=-origin)
                atom.rotate(axis, theta)
                atom.translate(vec=origin)

            else:
                atom.rotate(axis, theta)

        return None

    @requires_atoms()
    def print_xyz_file(self, title_line='', filename=None):
        """Print a standard xyz file from the Molecule's atoms"""

        if filename is None:
            filename = f'{self.name}.xyz'

        with open(filename, 'w') as xyz_file:
            print(self.n_atoms, title_line, sep='\n', file=xyz_file)
            for atom in self.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3}{x:^10.5f}{y:^10.5f}{z:^10.5f}', file=xyz_file)

        return None

    @requires_atoms()
    def get_coordinates(self):
        """Return a np.ndarray of size n_atoms x 3 containing the xyz coordinates of the molecule"""
        return np.array([atom.coord for atom in self.atoms])

    @requires_atoms()
    def single_point(self, method):
        """Calculate the single point energy of the species with a autode.wrappers.base.ElectronicStructureMethod"""
        logger.info(f'Running single point energy evaluation of {self.name}')

        sp = Calculation(name=f'{self.name}_sp', molecule=self, method=method,
                         keywords_list=method.keywords.sp, n_cores=Config.n_cores)
        sp.run()
        self.energy = sp.get_energy()

        return None

    def set_atoms(self, atoms):
        """Set the atoms of this species and from those the number of atoms"""

        self.atoms = atoms
        self.n_atoms = 0 if atoms is None else len(atoms)

        return None

    def set_coordinates(self, coords):
        """For coordinates as a np.ndarray with shape Nx3 set the coordinates of each atom"""

        assert coords.shape == (self.n_atoms, 3)

        for i, coord in enumerate(coords):
            self.atoms[i].coord = coord

        return None

    def __init__(self, name, atoms, charge, mult, solvent_name=None):
        """
        A molecular species. A collection of atoms with a charge and spin multiplicity in a solvent (None is gas phase)

        Arguments:
            name (str): Name of the species
            atoms (list(autode.atoms.Atom)): List of atoms in the species, or None
            charge (int): Charge on the species
            mult (int): Spin multiplicity of the species. 2S+1, where S is the number of unpaired electrons

        Keyword Arguments:
            solvent_name (str): Name of the solvent_name, or None
        """
        self.name = name

        self.atoms = atoms
        self.n_atoms = 0 if atoms is None else len(atoms)
        self.charge = charge
        self.mult = mult

        self.solvent = get_solvent(solvent_name=solvent_name) if solvent_name is not None else None

        self.energy = None                                               # Total electronic energy in Hartrees (float)

        self.charges = None                                              # List of partial atomic charges (list(float))

        self.graph = None                                                # NetworkX.Graph object with atoms and bonds


class SolvatedSpecies(Species):

    def single_point(self, method):
        logger.info(f'Running single point energy evaluation of {self.name}')

        point_charges = []
        for i, xyz in enumerate(self.mm_solvent_xyzs):
            point_charges.append(xyz + [self.solvent.charges[i % self.solvent.n_atoms]])

        sp = Calculation(name=self.name + '_sp', molecule=self, method=method, keywords_list=method.sp_keywords,
                         n_cores=Config.n_cores)
        sp.run()
        self.energy = sp.get_energy()

    def __int__(self, name, atoms, charge, mult, solvent_name):
        super(SolvatedSpecies, self).__init__(name, atoms, charge, mult, solvent_name)

        self.qm_solvent_xyzs = None
        self.mm_solvent_xyzs = None
