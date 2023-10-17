from copy import deepcopy
import numpy as np
from itertools import product as iterprod
from typing import Optional, Union, List, Sequence, TYPE_CHECKING

from autode.atoms import Atom, Atoms
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.geom import get_points_on_sphere
from autode.solvent.solvents import get_solvent
from autode.mol_graphs import union
from autode.species.species import Species
from autode.utils import requires_atoms, work_in
from autode.config import Config
from autode.methods import get_lmethod
from autode.conformers import Conformer
from autode.exceptions import MethodUnavailable

if TYPE_CHECKING:
    from autode.values import Angle
    from autode.solvent import Solvent


def get_complex_conformer_atoms(molecules, rotations, points):
    """
    Generate a conformer of a complex given a set of molecules, rotations for
    each and points on which to shift

    -----------------------------------------------------------------------
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
        theta, axis = np.random.uniform(-np.pi, np.pi), np.random.uniform(
            -1, 1, size=3
        )
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
    def __init__(
        self,
        *args: Species,
        name: str = "complex",
        do_init_translation: bool = False,
        copy: bool = True,
        solvent_name: Optional[str] = None,
    ):
        """
        Molecular complex e.g. VdW complex of one or more Molecules

        -----------------------------------------------------------------------
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
        super().__init__(
            name=name,
            atoms=sum(
                (deepcopy(mol.atoms) if copy else mol.atoms for mol in args),
                None,
            ),  # type: ignore
            charge=sum(mol.charge for mol in args),
            mult=sum(m.mult for m in args) - (len(args) - 1),
        )

        self._molecules = list(args)

        if do_init_translation:
            self._init_translation()

        self.solvent = self._init_solvent(solvent_name)
        self.graph = union(graphs=[mol.graph for mol in self._molecules])

    def __repr__(self):
        return self._repr(prefix="Complex")

    def __eq__(self, other):
        """Equality of two complexes"""
        return isinstance(other, self.__class__) and all(
            a == b for (a, b) in zip(self._molecules, other._molecules)
        )

    @Species.atoms.setter
    def atoms(self, value: Union[List[Atom], Atoms, None]):
        if value is None:
            self.graph = None
            self._molecules.clear()

        elif self.n_atoms != len(value):
            raise ValueError(
                f"Cannot set atoms in {self.name} with a "
                "different number of atoms. Molecular composition"
                " must have changed."
            )

        logger.warning(
            f"Modifying the atoms of {self.name} - assuming the "
            f"same molecular composition"
        )
        return super(Complex, type(self)).atoms.fset(self, value)

    @property
    def n_molecules(self) -> int:
        """Number of molecules in this molecular complex"""
        return len(self._molecules)

    def atom_indexes(self, mol_index: int):
        """
        List of atom indexes of a molecule withibn a Complex

        -----------------------------------------------------------------------
        Arguments:
            mol_index (int): Index of the molecule
        """
        if mol_index not in set(range(self.n_molecules)):
            raise AssertionError(
                f"Could not get idxs for molecule {mol_index}"
                f". Not present in this complex"
            )

        first_index = sum([mol.n_atoms for mol in self._molecules[:mol_index]])
        last_index = sum(
            [mol.n_atoms for mol in self._molecules[: mol_index + 1]]
        )

        return list(range(first_index, last_index))

    def reorder_atoms(self, mapping: dict) -> None:
        """
        Reorder the atoms in this complex using a dictionary keyed with current
        atom indexes and values as their new positions

        -----------------------------------------------------------------------
        Arguments:
            mapping (dict):
        """
        logger.warning(
            f"Reordering the atoms in a complex ({self.name}) will"
            f" not preserve the molecular composition"
        )

        return super().reorder_atoms(mapping)

    def _generate_conformers(self):
        """
        Generate rigid body conformers of a complex by (1) Fixing the first m
        olecule, (2) initialising the second molecule's COM evenly on the points
        of a sphere around the first with a random rotation and (3) iterating
        until all molecules in the complex have been added
        """
        n = self.n_molecules

        if n < 2:
            # Single (or zero) molecule complex only has a single *rigid body*
            # conformer
            self.conformers = [Conformer(name=self.name, species=self)]
            return None

        self.conformers = []
        m = 0  # Current conformer number

        points_on_sphere = get_points_on_sphere(
            n_points=Config.num_complex_sphere_points
        )

        for _ in iterprod(
            range(Config.num_complex_random_rotations), repeat=n - 1
        ):
            # Generate the rotation thetas and axes
            rotations = [
                np.random.uniform(-np.pi, np.pi, size=4) for _ in range(n - 1)
            ]

            for points in iterprod(points_on_sphere, repeat=n - 1):
                conf = Conformer(
                    name=f"{self.name}_conf{m}",
                    charge=self.charge,
                    mult=self.mult,
                )
                conf.solvent = self.solvent
                conf.atoms = get_complex_conformer_atoms(
                    self._molecules, rotations, points
                )
                self.conformers.append(conf)
                m += 1

                if m == Config.max_num_complex_conformers:
                    logger.warning(
                        f"Generated the maximum number of complex "
                        f"conformers ({m})"
                    )
                    return None

        logger.info(f"Generated {m} conformers")
        return None

    @work_in("conformers")
    def populate_conformers(self):
        r"""
        Generate and optimise with a low level method a set of conformers, the
        number of which is::

        Config.num_complex_sphere_points ×  Config.num_complex_random_rotations
         ^ (n molecules in complex - 1)

        This will not be exact as get_points_on_sphere does not return quite
        the desired number of points for small N.
        """
        n_confs = (
            Config.num_complex_sphere_points
            * Config.num_complex_random_rotations
            * (self.n_molecules - 1)
        )
        logger.info(
            f"Generating and optimising {n_confs} conformers of "
            f"{self.name} with a low-level method"
        )

        self._generate_conformers()

        try:
            lmethod = get_lmethod()
            for conformer in self.conformers:
                conformer.optimise(method=lmethod)
                conformer.print_xyz_file()

        except MethodUnavailable:
            logger.error("Could not optimise complex conformers")

        return None

    def translate_mol(self, vec: Sequence[float], mol_index: int):
        """
        Translate a molecule within a complex by a vector

        -----------------------------------------------------------------------
        Arguments:
            vec (np.ndarray | list(float)): Length 3 vector

            mol_index (int): Index of the molecule to translate. e.g. 2 will
                             translate molecule 1 in the complex
                             they are indexed from 0
        """
        logger.info(
            f"Translating molecule {mol_index} by {vec} in {self.name}"
        )

        if mol_index not in set(range(self.n_molecules)):
            raise ValueError(
                f"Could not translate molecule {mol_index} "
                "not present in this complex"
            )

        for atom_idx in self.atom_indexes(mol_index):
            self.atoms[atom_idx].translate(vec)

        return None

    def rotate_mol(
        self,
        axis: Union[np.ndarray, Sequence],
        theta: Union["Angle", float],
        mol_index: int,
        origin: Union[np.ndarray, Sequence, None] = None,
    ):
        """
        Rotate a molecule within a complex an angle theta about an axis given
        an origin

        -----------------------------------------------------------------------
        Arguments:
            axis (np.ndarray | list): Length 3 vector

            theta (float | autode.values.Angle):

            origin (np.ndarray | list): Length 3 vector

            mol_index (int): Index of the molecule to translate. e.g. 2 will
                            translate molecule 1 in the complex
                             they are indexed from 0
        """
        logger.info(
            f"Rotating molecule {mol_index} by {theta:.4f} radians "
            f"in {self.name}"
        )

        if mol_index not in set(range(self.n_molecules)):
            raise ValueError(
                f"Could not rotate molecule {mol_index} "
                "not present in this complex"
            )

        for atom_idx in self.atom_indexes(mol_index):
            self.atoms[atom_idx].rotate(axis, theta, origin)

        return None

    @requires_atoms
    def calc_repulsion(self, mol_index: int):
        """Calculate the repulsion between a molecule and the rest of the
        complex"""

        coords = self.coordinates

        mol_indexes = self.atom_indexes(mol_index)
        mol_coords = [coords[i] for i in mol_indexes]
        other_coords = [
            coords[i] for i in range(self.n_atoms) if i not in mol_indexes
        ]

        # Repulsion is the sum over all pairs 1/r^4
        distance_mat = distance_matrix(mol_coords, other_coords)
        repulsion = 0.5 * np.sum(np.power(distance_mat, -4))

        return repulsion

    def _init_translation(self):
        """Translate all molecules initially to avoid overlaps"""

        if self.n_molecules < 2:
            return  # No need to translate 0 or 1 molecule

        # Points on the unit sphere maximally displaced from one another
        points = get_points_on_sphere(n_points=self.n_molecules)

        # Shift along the vector defined on the unit sphere by the molecule's
        # radius + 4Å, which should generate a somewhat reasonable geometry
        for i in range(self.n_molecules):
            self.translate_mol(
                vec=(self._molecules[i].radius + 4) * points[i], mol_index=i
            )
        return None

    def _init_solvent(
        self, solvent_name: Optional[str]
    ) -> Optional["Solvent"]:
        """Initial solvent"""

        if solvent_name is not None:
            return get_solvent(solvent_name, kind="implicit")

        if self.n_molecules > 0:
            solvent = self._molecules[0].solvent
            if any(solvent != mol.solvent for mol in self._molecules):
                raise AssertionError(
                    "Cannot form a complex with molecules in "
                    "different solvents"
                )

            return solvent

        return None


class ReactantComplex(Complex):
    # NOTE: Methods must be identical to ProductComplex

    def to_product_complex(self):
        """Return a product complex from this reactant complex"""

        prod_complex = self.copy()
        prod_complex.__class__ = ProductComplex

        return prod_complex

    def __init__(self, *args, name="reac_complex", **kwargs):
        """
        Reactant complex

        -----------------------------------------------------------------------
        Arguments:
            *args (autode.species.Reactant):

        Keyword Arguments:
            name (str):
        """
        super().__init__(*args, name=name, **kwargs)


class ProductComplex(Complex):
    # NOTE: Methods must be identical to ReactantComplex

    def to_reactant_complex(self):
        """Return a reactant complex from this product complex"""

        reac_complex = self.copy()
        reac_complex.__class__ = ReactantComplex

        return reac_complex

    def __init__(self, *args, name="prod_complex", **kwargs):
        """
        Product complex

        -----------------------------------------------------------------------
        Arguments:
            *args (autode.species.Product):

        Keyword Arguments:
            name (str):
        """
        super().__init__(*args, name=name, **kwargs)


class NCIComplex(Complex):
    """Non covalent interaction complex"""
