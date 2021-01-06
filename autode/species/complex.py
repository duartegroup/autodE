from copy import deepcopy
import numpy as np
from itertools import product as iterprod
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.geom import get_points_on_sphere
from autode.mol_graphs import union
from autode.species.species import Species
from autode.utils import requires_atoms
from autode.config import Config
from autode.methods import get_lmethod
from autode.conformers.conformer import get_conformer
from autode.exceptions import MethodUnavailable


def get_complex_conformer_atoms(molecules, rotations, points):
    """
    Generate a conformer of a complex given a set of molecules, rotations for
    each and points on which to shift

    Arguments:
        molecules (list(autode.species.Species)):
        rotations (list(np.ndarray)): List of len 4 np arrays containing the
                  [theta, x, y, z] defining the rotation
                                      amount and axis
        points: (list(np.ndarray)): List of length 3 np arrays containing the
        point to add the molecule with index i

    Returns:
        (list(autode.atoms.Atom))
    """
    assert len(molecules) - 1 == len(rotations) == len(points) > 0

    # First molecule is static so start with those atoms
    atoms = deepcopy(molecules[0].atoms)

    # For each molecule add it to the current set of atoms with the centroid
    # ~ COM located at the origin
    for i, molecule in enumerate(molecules[1:]):

        centroid = np.average(np.array([atom.coord for atom in atoms]), axis=0)

        # Shift to the origin and rotate randomly, by the same amount
        theta, axis = np.random.uniform(-np.pi, np.pi), np.random.uniform(-1, 1, size=3)
        for atom in atoms:
            atom.translate(vec=-centroid)
            atom.rotate(axis, theta)

        coords = np.array([atom.coord for atom in atoms])

        mol_centroid = np.average(molecule.get_coordinates(), axis=0)
        shifted_mol_atoms = deepcopy(molecule.atoms)

        # Shift the molecule to the origin then rotate randomly
        theta, axis = rotations[i][0], rotations[i][1:]
        for atom in shifted_mol_atoms:
            atom.translate(vec=-mol_centroid)
            atom.rotate(axis, theta)

        # Shift until the current molecules don't overlap with the current
        #  atoms, i.e. aren't far enough apart
        far_enough_apart = False

        # Shift the molecule by 0.1 Å in the direction of the point
        # (which has length 1) until the
        # minimum distance to the rest of the complex is 2.0 Å
        while not far_enough_apart:

            for atom in shifted_mol_atoms:
                atom.coord += points[i] * 0.1

            mol_coords = np.array([atom.coord for atom in shifted_mol_atoms])

            if np.min(distance_matrix(coords, mol_coords)) > 2.0:
                far_enough_apart = True

        atoms += shifted_mol_atoms

    return atoms


class Complex(Species):


    def get_atom_indexes(self, mol_index):
        """Get the first and last atom indexes of a molecule in a Complex"""
        assert mol_index < len(self.molecules)

        first_index = sum([mol.n_atoms for mol in self.molecules[:mol_index]])
        last_index = sum([mol.n_atoms for mol in self.molecules[:mol_index + 1]])

        return list(range(first_index, last_index))

    def _generate_conformers(self):
        """
        Generate rigid body conformers of a complex by (1) Fixing the first m
        olecule, (2) initialising the second molecule's COM evenly on the points
        of a sphere around the first with a random rotation and (3) iterating
        until all molecules in the complex have been added
        """
        if len(self.molecules) < 2:
            # Single (or zero) molecule complex only has a single *rigid body*
            # conformer
            self.conformers = [get_conformer(name=self.name, species=self)]

            return None

        n_molecules = len(self.molecules)  # Number of molecules in the complex
        self.conformers = []
        n = 0                              # Current conformer number

        points_on_sphere = get_points_on_sphere(n_points=Config.num_complex_sphere_points)

        for _ in iterprod(range(Config.num_complex_random_rotations), repeat=n_molecules-1):
            # Generate the rotation thetas and axes
            rotations = [np.random.uniform(-np.pi, np.pi, size=4) for _ in range(n_molecules - 1)]

            for points in iterprod(points_on_sphere, repeat=n_molecules-1):

                conformer = get_conformer(species=self, name=f'{self.name}_conf{n}')
                atoms = get_complex_conformer_atoms(self.molecules, rotations, points)
                conformer.atoms = atoms

                self.conformers.append(conformer)
                n += 1

                if n == Config.max_num_complex_conformers:
                    logger.warning(f'Generated the maximum number of complex conformers ({n})')
                    return None

        logger.info(f'Generated {n} conformers')
        return None

    def populate_conformers(self):
        """
        Generate and optimise with a low level method a set of conformers, the
        number of which is
        Config.num_complex_sphere_points ×  Config.num_complex_random_rotations
         ^ (n molecules in complex - 1)
        """
        n_confs = Config.num_complex_sphere_points * Config.num_complex_random_rotations * (len(self.molecules) - 1 )
        logger.info(f'Generating and optimising {n_confs} conformers of {self.name}')

        self._generate_conformers()

        try:
            lmethod = get_lmethod()
            for conformer in self.conformers:
                conformer.optimise(method=lmethod)
                conformer.print_xyz_file()

        except MethodUnavailable:
            logger.error('Could not optimise complex conformers')

        return None

    @requires_atoms()
    def translate_mol(self, vec, mol_index):
        """
        Translate a molecule within a complex by a vector

        Arguments:
            vec (np.ndarray): Length 3 vector
            mol_index (int): Index of the molecule to translate. e.g. 2 will
                             translate molecule 1 in the complex
                             they are indexed from 0

        """
        logger.info(f'Translating molecule {mol_index} by {vec} in {self.name}')

        for atom_index in self.get_atom_indexes(mol_index):
            self.atoms[atom_index].translate(vec)

        return None

    @requires_atoms()
    def rotate_mol(self, axis, theta, mol_index, origin=np.zeros(3)):
        """
        Rotate a molecule within a complex an angle theta about an axis given
        an origin

        Arguments:
            axis (np.ndarray): Length 3 vector
            theta (float): Length 3 vector
            origin (np.ndarray): Length 3 vector
            mol_index (int): Index of the molecule to translate. e.g. 2 will
                            translate molecule 1 in the complex
                             they are indexed from 0

        """
        logger.info(f'Rotating molecule {mol_index} by {theta:.4f} radians in {self.name}')

        for atom_index in self.get_atom_indexes(mol_index):
            self.atoms[atom_index].translate(vec=-origin)
            self.atoms[atom_index].rotate(axis, theta)
            self.atoms[atom_index].translate(vec=origin)

        return None

    @requires_atoms()
    def calc_repulsion(self, mol_index):
        """Calculate the repulsion between a molecule and the rest of the
        complex"""

        coordinates = self.get_coordinates()

        mol_indexes = self.get_atom_indexes(mol_index)
        mol_coords = [coordinates[i] for i in mol_indexes]
        other_coords = [coordinates[i] for i in range(self.n_atoms) if i not in mol_indexes]

        # Repulsion is the sum over all pairs 1/r^4
        distance_mat = distance_matrix(mol_coords, other_coords)
        repulsion = 0.5 * np.sum(np.power(distance_mat, -4))

        return repulsion

    def __init__(self, *args, name='complex'):
        """
        Molecular complex e.g. VdW complex of one or more Molecules

        Arguments:
            *args (autode.species.Species):

        Keyword Arguments:
            name (str):
        """
        self.molecules = args
        self.molecule_atom_indexes = []

        # Calculate the overall charge and spin multiplicity on the system and
        # initialise
        complex_charge = sum([mol.charge for mol in self.molecules])
        complex_mult = sum([mol.mult for mol in self.molecules]) - (len(self.molecules) - 1)

        complex_atoms = []
        for mol in self.molecules:
            complex_atoms += deepcopy(mol.atoms)

        super().__init__(name=name, atoms=complex_atoms, charge=complex_charge, mult=complex_mult)

        self.solvent = self.molecules[0].solvent if len(self.molecules) > 0 else None

        self.graph = union(graphs=[mol.graph for mol in self.molecules])


class ReactantComplex(Complex):
    pass


class ProductComplex(Complex):
    pass


class SolvatedReactantComplex(Complex):

    def run_const_opt(self, const_opt):
        """Run a constrained optimisation of the ReactantComplex"""
        raise NotImplementedError

    def __init__(self, solvent_mol, *args, name='complex'):
        super().__init__(*args, name=name)
        self.solvent_mol = solvent_mol
        self.qm_solvent_atoms = None
        self.mm_solvent_atoms = None


class NCIComplex(Complex):
    pass


def is_solvated_reactant_complex(molecule_complex):
    return isinstance(molecule_complex, SolvatedReactantComplex)


def get_complexes(reaction):
    """Creates Reactant and Product complexes for the reaction"""

    if reaction.reacs[0].is_explicitly_solvated():
        raise NotImplementedError

    reac = ReactantComplex(*reaction.reacs, name=f'{str(reaction)}_reactant')
    prod = ProductComplex(*reaction.prods, name=f'{str(reaction)}_product')

    return reac, prod
