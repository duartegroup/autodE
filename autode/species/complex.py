from copy import deepcopy
import numpy as np
from typing import Optional, Union, List
from autode.atoms import Atom, Atoms
from itertools import product as iterprod
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.geom import get_points_on_sphere
from autode.solvent.solvents import get_solvent
from autode.mol_graphs import union
from autode.species.species import Species
from autode.utils import requires_atoms, work_in
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

        mol_centroid = np.average(molecule.coordinates, axis=0)
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

    def __repr__(self):
        return self._repr(prefix='Complex')

    @property
    def n_molecules(self) -> int:
        """Number of molecules in this molecular complex"""
        return len(self.molecules)

    @property
    def charge(self) -> int:
        """
        Total charge on the complex

        Returns:
            (int): units of e
        """
        return sum([mol.charge for mol in self.molecules])

    @charge.setter
    def charge(self, _):
        raise ValueError('Cannot set the charge of a complex. Ambiguous on '
                         'which molecule the charge should be. Consider:'
                         'complex.to_species() then setting the charge')

    @property
    def mult(self) -> int:
        """
        Total spin multiplicity on the complex

        Returns:
            (int): 2S + 1, where S is the number of unpaired electrons
        """
        return sum([m.mult for m in self.molecules]) - (self.n_molecules - 1)

    @mult.setter
    def mult(self, _):
        raise ValueError('Cannot set the spin multiplicity of a complex. '
                         'Ambiguous on which molecule the unpaired electron(s)'
                         ' are located. Consider: complex.to_species() then '
                         'setting the mult')

    @property
    def solvent(self) -> 'autode.solvent.Solvent':

        if not hasattr(self, 'molecules') or self.n_molecules == 0:
            return None

        return self.molecules[0].solvent

    @solvent.setter
    def solvent(self, value: Optional['autode.solvent.Solvent']):

        if hasattr(self, 'molecules'):
            for mol in self.molecules:
                mol.solvent = value

    @property
    def atoms(self) -> Optional[Atoms]:
        """
        All atoms in this complex, and None if there are no molecules

        Returns:
            (autode.atoms.Atoms | None)
        """
        all_atoms = Atoms()

        for mol in self.molecules:
            if mol.atoms is not None:
                all_atoms += mol.atoms

        return all_atoms if len(all_atoms) > 0 else None

    @atoms.setter
    def atoms(self, value: Union[List[Atom], Atoms, None]):
        """
        Set the atoms of this complex

        Arguments:
            value (list(Atom) | Atoms | None):
        """
        if self.n_molecules != 0 and value is None:
            raise ValueError(f'Could not set the Atoms of {self.n_molecules} '
                             f'as None.')

        if len(value) != self.n_atoms:
            raise ValueError(f'Could not set the atoms. Needed '
                             f'{self.n_atoms} but had {len(value)}')

        for i, mol in enumerate(self.molecules):
            mol.atoms = Atoms([value[idx] for idx in self.atom_indexes(i)])

    def to_species(self) -> Species:
        """
        Convert this complex into a species, will loose the molecular
        composition

        Returns:
            (autode.species.Species):
        """
        if self.n_molecules == 0 or self.atoms is None:
            raise ValueError(f'Could not convert {self.name} into a species '
                             f'had no atoms')

        species = Species(name=self.name, atoms=self.atoms.copy(),
                          charge=self.charge, mult=self.mult)
        species.solvent = self.solvent
        species.graph = self.graph

        return species

    def atom_indexes(self, mol_index):
        """Get the first and last atom indexes of a molecule in a Complex"""
        if mol_index not in set(range(self.n_molecules)):
            raise AssertionError(f'Could not get idxs for molecule {mol_index}'
                                 f'. Not present in this complex')

        first_index = sum([mol.n_atoms for mol in self.molecules[:mol_index]])
        last_index = sum([mol.n_atoms for mol in self.molecules[:mol_index + 1]])

        return list(range(first_index, last_index))

    def reorder_atoms(self, mapping: dict) -> None:
        """
        Reorder the atoms in this complex

        Arguments:
            mapping (dict):
        """
        raise NotImplementedError

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

    @work_in('conformers')
    def populate_conformers(self):
        """
        Generate and optimise with a low level method a set of conformers, the
        number of which is
        Config.num_complex_sphere_points ×  Config.num_complex_random_rotations
         ^ (n molecules in complex - 1)
        """
        n_confs = (Config.num_complex_sphere_points
                   * Config.num_complex_random_rotations
                   * (len(self.molecules) - 1))
        logger.info(f'Generating and optimising {n_confs} conformers of '
                    f'{self.name} with a low-level method')

        self._generate_conformers()

        try:
            lmethod = get_lmethod()
            for conformer in self.conformers:
                conformer.optimise(method=lmethod)
                conformer.print_xyz_file()

        except MethodUnavailable:
            logger.error('Could not optimise complex conformers')

        return None

    def translate_mol(self, vec, mol_index):
        """
        Translate a molecule within a complex by a vector

        Arguments:
            vec (np.ndarray | list): Length 3 vector

            mol_index (int): Index of the molecule to translate. e.g. 2 will
                             translate molecule 1 in the complex
                             they are indexed from 0

        """
        logger.info(f'Translating molecule {mol_index} by {vec} in {self.name}')

        if mol_index not in set(range(self.n_molecules)):
            raise AssertionError(f'Could not translate molecule {mol_index} '
                                 f'not present in this complex')

        self.molecules[mol_index].translate(vec)
        return None

    def rotate_mol(self, axis, theta, mol_index, origin=np.zeros(3)):
        """
        Rotate a molecule within a complex an angle theta about an axis given
        an origin

        Arguments:
            axis (np.ndarray | list): Length 3 vector

            theta (float | autode.values.Angle):

            origin (np.ndarray | list): Length 3 vector

            mol_index (int): Index of the molecule to translate. e.g. 2 will
                            translate molecule 1 in the complex
                             they are indexed from 0
        """
        logger.info(f'Rotating molecule {mol_index} by {theta:.4f} radians '
                    f'in {self.name}')

        if mol_index not in set(range(self.n_molecules)):
            raise AssertionError(f'Could not rotate molecule {mol_index} '
                                 f'not present in this complex')

        self.molecules[mol_index].rotate(axis, theta, origin)
        return None

    @requires_atoms
    def calc_repulsion(self, mol_index):
        """Calculate the repulsion between a molecule and the rest of the
        complex"""

        coords = self.coordinates

        mol_indexes = self.atom_indexes(mol_index)
        mol_coords = [coords[i] for i in mol_indexes]
        other_coords = [coords[i] for i in range(self.n_atoms)
                        if i not in mol_indexes]

        # Repulsion is the sum over all pairs 1/r^4
        distance_mat = distance_matrix(mol_coords, other_coords)
        repulsion = 0.5 * np.sum(np.power(distance_mat, -4))

        return repulsion

    def _init_translation(self):
        """Translate all molecules initially to avoid overlaps"""

        if self.n_molecules < 2:
            return   # No need to translate 0 or 1 molecule

        # Points on the unit sphere maximally displaced from one another
        points = get_points_on_sphere(n_points=self.n_molecules)

        # Shift along the vector defined on the unit sphere by the molecule's
        # radius + 4Å, which should generate a somewhat reasonable geometry
        for i in range(self.n_molecules):
            self.translate_mol(vec=(self.molecules[i].radius + 4) * points[i],
                               mol_index=i)
        return None

    def _init_solvent(self, solvent_name):
        """Initial solvent"""

        if solvent_name is not None:
            return get_solvent(solvent_name)

        if self.n_molecules > 0:
            if any(self.molecules[0].solvent != mol.solvent for mol in self.molecules):
                raise AssertionError('Cannot form a complex with molecules in '
                                     'different solvents')

            return self.molecules[0].solvent

        return None

    def __init__(self,
                 *args:                Species,
                 name:                 str = 'complex',
                 do_init_translation:  bool = False,
                 copy:                 bool = True,
                 solvent_name: Optional[str] = None):
        """
        Molecular complex e.g. VdW complex of one or more Molecules

        Arguments:
            *args (autode.species.Species):

        Keyword Arguments:
            name (str):

            do_init_translation (bool): Translate molecules initially such
                                        that they donot overlap

            copy (bool): Should the molecules be copied into this complex?

            solvent_name (str | None): Name of the solvent, if None then select
                                       the first solvent from the constituent
                                       molecules
        """
        super().__init__(name=name, atoms=None, charge=0, mult=1)

        self.molecules = [mol.copy() if copy else mol for mol in args]

        if do_init_translation:
            self._init_translation()

        self.solvent = self._init_solvent(solvent_name)
        self.graph = union(graphs=[mol.graph for mol in self.molecules])


class ReactantComplex(Complex):

    def to_product_complex(self):
        """Return a product complex from this reactant complex"""

        prod_complex = self.copy()
        prod_complex.__class__ = ProductComplex

        return prod_complex

    def __init__(self, *args, name='reac_complex', **kwargs):
        """
        Reactant complex

        Arguments:
            *args (autode.species.Reactant):

        Keyword Arguments:
            name (str):
        """
        super().__init__(*args, name=name, **kwargs)


class ProductComplex(Complex):

    def to_reactant_complex(self):
        """Return a reactant complex from this product complex"""

        reac_complex = self.copy()
        reac_complex.__class__ = ReactantComplex

        return reac_complex

    def __init__(self, *args, name='prod_complex', **kwargs):
        """
        Product complex

        Arguments:
            *args (autode.species.Product):

        Keyword Arguments:
            name (str):
        """
        super().__init__(*args, name=name, **kwargs)


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
