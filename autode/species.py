import numpy as np
from autode.log import logger
from autode.solvent.solvents import get_solvent
from autode.calculation import Calculation
from autode import mol_graphs
from autode.methods import get_lmethod
from autode.conformers.conformers import get_unique_confs
from autode.config import Config
from autode.utils import requires_atoms
from autode.utils import requires_conformers


class Species:

    def _generate_conformers(self, *args, **kwargs):
        raise NotImplemented

    @requires_conformers()
    def _set_lowest_energy_conformer(self):
        """Set the species energy and atoms as those of the lowest energy conformer"""

        lowest_energy = None
        for conformer in self.conformers:
            if conformer.energy is None:
                continue

            # Conformers don't have a molecular graph, so make it
            mol_graphs.make_graph(conformer)

            if not mol_graphs.is_isomorphic(self.graph, conformer.graph, ignore_active_bonds=True):
                logger.warning('Conformer had a different molecular graph. Ignoring')
                continue

            # If the conformer retains the same connectivity, up the the active atoms in the species graph

            if lowest_energy is None:
                lowest_energy = conformer.energy

            if conformer.energy <= lowest_energy:
                self.energy = conformer.energy
                self.set_atoms(atoms=conformer.atoms)
                lowest_energy = conformer.energy

        return None

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

    @requires_atoms()
    def get_distance(self, atom_i, atom_j):
        """Get the distance between two atoms in the species"""
        return np.linalg.norm(self.atoms[atom_i].coord - self.atoms[atom_j].coord)

    def find_lowest_energy_conformer(self, low_level_method=None, high_level_method=None):
        """
        For a molecule object find the lowest conformer in energy and set the molecule.atoms and molecule.energy

        Arguments:
            low_level_method (autode.wrappers.ElectronicStructureMethod):
            high_level_method (autode.wrappers.ElectronicStructureMethod):
        """
        logger.info('Finding lowest energy conformer')

        if low_level_method is None:
            logger.info('Getting the default low level method')
            low_level_method = get_lmethod()

        try:
            self._generate_conformers()
        except NotImplementedError:
            logger.error('Could not generate conformers. _generate_conformers() not implemented')
            return None

        # For all the generated conformers optimise with the low level of theory
        for i in range(len(self.conformers)):
            self.conformers[i].optimise(low_level_method)

        # Strip conformers that are similar based on an energy criteria or don't have an energy
        self.conformers = get_unique_confs(conformers=self.conformers)

        if high_level_method is not None:
            # Re-optimise all the conformers with the higher level of theory to get more accurate energies
            [self.conformers[i].optimise(high_level_method) for i in range(len(self.conformers))]

        self._set_lowest_energy_conformer()

        logger.info(f'Lowest energy conformer found. E = {self.energy}')
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

        self.energy = None                                              # Total electronic energy in Hartrees (float)

        self.charges = None                                             # List of partial atomic charges (list(float))

        self.graph = None                                               # NetworkX.Graph object with atoms and bonds

        self.conformers = None                                          # List of autode.conformers.conformers.Conformer


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
