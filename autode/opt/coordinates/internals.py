"""
Internal coordinates. Notation follows:


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
G : Spectroscopic G matrix

Set-up of redundant primitives is based on J. Chem. Phys., 117, 2002, 9160
"""
import numpy as np
import itertools
from typing import Any, Optional, Type, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from autode.values import Angle, Distance
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.coordinates.primitives import (
    PrimitiveInverseDistance,
    Primitive,
    PrimitiveDistance,
    ConstrainedPrimitiveDistance,
    PrimitiveBondAngle,
    PrimitiveDummyLinearAngle,
    PrimitiveLinearAngle,
    PrimitiveDihedralAngle,
    LinearBendType,
)

if TYPE_CHECKING:
    from autode.species import Species
    from autode.opt.coordinates.cartesian import CartesianCoordinates
    from autode.opt.coordinates.primitives import (
        ConstrainedPrimitive,
        _DistanceFunction,
    )


# Angle threshold for linearity (should be in radians)
_lin_thresh = Angle(170, "deg").to("rad")


class InternalCoordinates(OptCoordinates, ABC):  # lgtm [py/missing-equals]
    def __new__(cls, input_array) -> "InternalCoordinates":
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array, units="Å")

        arr._x = None
        arr.primitives = None

        for attr in ("_x", "primitives"):
            setattr(arr, attr, getattr(input_array, attr, None))

        return arr

    def __array_finalize__(self, obj: "OptCoordinates") -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""
        OptCoordinates.__array_finalize__(self, obj)

        for attr in ("_x", "primitives"):
            setattr(self, attr, getattr(obj, attr, None))

        return

    @property
    def n_constraints(self) -> int:
        """Number of constraints in these coordinates"""
        return self.primitives.n_constrained

    @property
    def constrained_primitives(self) -> List["ConstrainedPrimitive"]:
        return [p for p in self.primitives if p.is_constrained]

    @property
    def n_satisfied_constraints(self) -> int:
        """Number of constraints that are satisfied in these coordinates"""
        x = self.to("cartesian")
        return sum(p.is_satisfied(x) for p in self.constrained_primitives)


class PIC(list, ABC):
    """Primitive internal coordinates"""

    def __init__(self, *args: Any):
        """
        List of primitive internal coordinates with a Wilson B matrix.
        If there are no arguments then all possible primitive coordinates
        will be generated
        """
        super().__init__(args)

        self._B: Optional[np.ndarray] = None

        if not self._are_all_primitive_coordinates(args):
            raise ValueError(
                "Cannot construct primitive internal coordinates "
                f"from {args}. Must be primitive internals"
            )

    def add(self, item: Primitive) -> None:
        """Add a primitive to this set of primitive coordinates"""
        assert isinstance(item, Primitive), "Must be a primitive"
        # prevent duplication of primitives
        if item not in self:
            super().append(item)

    def append(self, item: Primitive) -> None:
        """Append an item to this set of primitives"""
        raise NotImplementedError(
            "Please use PIC.add() to add new primitives to the set"
        )

    @property
    def B(self) -> np.ndarray:
        """Wilson B matrix"""

        if self._B is None:
            raise AttributeError(
                f"{self} had no B matrix. Please calculate "
                f"the value of the primitives to determine B"
            )

        return self._B

    @property
    def G(self) -> np.ndarray:
        """Spectroscopic G matrix as the symmetrised Wilson B matrix"""
        return np.dot(self.B, self.B.T)

    @classmethod
    def from_cartesian(
        cls,
        x: "CartesianCoordinates",
    ) -> "PIC":
        """Construct a complete set of primitive internal coordinates from
        a set of Cartesian coordinates"""

        pic = cls()
        pic._populate_all(x=x)

        return pic

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Populate Primitive-s used in the construction of set"""

        q = self._calc_q(x)
        self._calc_B(x)

        return q

    def close_to(self, x: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        Calculate a set of primitive internal coordinates (PIC) that are
        'close to' another set. This means that the restriction on dihedral
        angles being in the range (-π, π] is relaxed in favour of the smallest
        ∆q possible (where q is a value of a primitive coordinate).
        """
        assert len(self) == len(other) and isinstance(other, np.ndarray)

        q = self._calc_q(x)
        self._calc_B(x)

        for i, primitive in enumerate(self):
            if isinstance(primitive, PrimitiveDihedralAngle):
                dq = q[i] - other[i]

                if np.abs(dq) > np.pi:  # Ensure |dq| < π
                    q[i] -= np.sign(dq) * 2 * np.pi

        return q

    def __eq__(self, other: Any):
        """Comparison of two PIC sets"""

        is_equal = (
            isinstance(other, PIC)
            and len(other) == len(self)
            and all(p0 == p1 for p0, p1 in zip(self, other))
        )

        return is_equal

    def _calc_q(self, x: np.ndarray) -> np.ndarray:
        """Calculate the value of the internals"""

        if len(self) == 0:
            self._populate_all(x)

        return np.array([q(x) for q in self])

    @abstractmethod
    def _populate_all(self, x: np.ndarray) -> None:
        """Populate primitives from an array of cartesian coordinates"""

    def _calc_B(self, x: np.ndarray) -> None:
        """Calculate the Wilson B matrix"""

        if len(self) == 0:
            raise ValueError(
                "Cannot calculate the Wilson B matrix, no "
                "primitive internal coordinates"
            )

        cart_coords = x.ravel()

        B = np.zeros(shape=(len(self), len(cart_coords)))

        for i, primitive in enumerate(self):
            B[i] = primitive.derivative(x=cart_coords)

        self._B = B
        return None

    @staticmethod
    def _are_all_primitive_coordinates(args: tuple) -> bool:
        return all(isinstance(arg, Primitive) for arg in args)

    @property
    def n_constrained(self) -> int:
        """Number of constrained primitive internal coordinates"""
        return sum(p.is_constrained for p in self)


class _FunctionOfDistances(PIC):
    @property
    @abstractmethod
    def _primitive_type(self) -> Type["_DistanceFunction"]:
        """Type of primitive coordinate defining f(r_ij)"""

    def _populate_all(self, x: np.ndarray):
        n_atoms = len(x.flatten()) // 3

        # Add all the unique inverse distances (i < j)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                self.add(self._primitive_type(i, j))

        return None


class PrimitiveInverseDistances(_FunctionOfDistances):
    """1 / r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return PrimitiveInverseDistance


class PrimitiveDistances(_FunctionOfDistances):
    """r_ij for all unique pairs i,j. Will be redundant"""

    @property
    def _primitive_type(self):
        return PrimitiveDistance


class AnyPIC(PIC):
    def _populate_all(self, x: np.ndarray) -> None:
        raise RuntimeError("Cannot populate all on an AnyPIC instance")

    @classmethod
    def from_species(cls, mol: "Species") -> "AnyPIC":
        """
        Build a set of primitives from the species, using the graph as
        a starting point for the connectivity of the species. Also joins
        any disjoint parts of the graph, and adds hydrogen bonds to
        ensure that the primitives are redundant.

        Args:
            mol: The species object

        Returns:
            (AnyPIC): The set of primitive internals
        """
        pic = cls()
        # take a copy of mol as mol.graph might be changed
        mol = mol.copy()
        _connect_graph_for_species(mol)
        pic._add_bonds_from_species(mol)
        pic._add_angles_from_species(mol)
        pic._add_dihedrals_from_species(mol)
        pic._add_chain_dihedrals_from_species(mol)
        return pic

    def _add_bonds_from_species(
        self,
        mol: "Species",
    ):
        """
        Add bonds to the current set of primitives, from the
        connectivity graph of the species

        Args:
            mol: The species object
        """
        assert mol.graph is not None

        n = 0
        for i, j in sorted(mol.graph.edges):
            if (
                mol.constraints.distance is not None
                and (i, j) in mol.constraints.distance
            ):
                r = mol.constraints.distance[(i, j)]
                self.add(ConstrainedPrimitiveDistance(i, j, r))
                n += 1
            else:
                self.add(PrimitiveDistance(i, j))
        assert n == mol.constraints.n_distance

        return None

    @staticmethod
    def _get_ref_for_linear_angle(
        mol,
        a,
        b,
        c,
        bonded: bool,
        dist_thresh=Distance(4, "ang"),
    ) -> Optional[int]:
        """
        Get a reference atom for describing a linear angle, which
        must not itself be linear to the atoms in the angle in
        any combination. The linear angle is a--b--c here.

        Args:
            mol:
            a:
            b:
            c:
            bonded: Whether to look for only atoms bonded to the central
                    atom (b) for reference
            dist_thresh: The distance threshold to check for atoms if
                    bonded = False

        Returns:
            (int|None): The index of the ref. atom if found, else None
        """

        # only check bonded atoms if requested
        if bonded:
            near_atoms = list(mol.graph.neighbors(b))
            near_atoms.remove(a)
            near_atoms.remove(c)

        # otherwise get all atoms in 4 A radius except a, b, c
        else:
            near_atoms = [
                idx
                for idx in range(mol.n_atoms)
                if mol.distance(b, idx) < dist_thresh and idx not in (a, b, c)
            ]

        # get atoms closest to perpendicular
        deviations_from_90 = {}
        for atom in near_atoms:
            i_b_a = mol.angle(atom, b, a)
            if i_b_a > _lin_thresh or i_b_a < (np.pi - _lin_thresh):
                continue
            i_b_c = mol.angle(atom, b, c)
            if i_b_c > _lin_thresh or i_b_c < (np.pi - _lin_thresh):
                continue
            deviation_a = abs(i_b_a - np.pi / 2)
            deviation_b = abs(i_b_c - np.pi / 2)
            avg_dev = (deviation_a + deviation_b) / 2
            deviations_from_90[atom] = avg_dev

        if len(deviations_from_90) == 0:
            return None

        return min(deviations_from_90, key=deviations_from_90.get)  # type: ignore

    def _add_angles_from_species(
        self,
        mol: "Species",
    ) -> None:
        """
        Modify the set of primitives in-place by adding angles, from the
        connectivity graph supplied

        Args:
            mol (Species): The species object
        """
        assert mol.graph is not None

        for o in range(mol.n_atoms):
            for n, m in itertools.combinations(mol.graph.neighbors(o), r=2):
                if mol.angle(m, o, n) < _lin_thresh:
                    self.add(PrimitiveBondAngle(m=m, o=o, n=n))
                else:
                    # If central atom is connected to another atom, then the
                    # linear angle is skipped and instead an out-of-plane
                    # (improper dihedral) coordinate is used
                    r = self._get_ref_for_linear_angle(
                        mol, m, o, n, bonded=True
                    )
                    if r is not None:
                        self.add(PrimitiveDihedralAngle(m, r, o, n))
                        continue

                    # Otherwise, we use a nearby (< 4.0 A) reference atom to
                    # define two orthogonal linear bends
                    r = self._get_ref_for_linear_angle(
                        mol, m, o, n, bonded=False
                    )
                    if r is not None:
                        self.add(
                            PrimitiveLinearAngle(
                                m, o, n, r, LinearBendType.BEND
                            )
                        )
                        self.add(
                            PrimitiveLinearAngle(
                                m, o, n, r, LinearBendType.COMPLEMENT
                            )
                        )

                    # For completely linear molecules (CO2), there will be no such
                    # reference atoms, so use dummy atoms instead
                    else:
                        self.add(
                            PrimitiveDummyLinearAngle(
                                m, o, n, LinearBendType.BEND
                            )
                        )
                        self.add(
                            PrimitiveDummyLinearAngle(
                                m, o, n, LinearBendType.COMPLEMENT
                            )
                        )

        return None

    def _add_dihedrals_from_species(
        self,
        mol: "Species",
    ) -> None:
        """
        Modify the set of primitives in-place by adding dihedrals (torsions),
        from the connectivity graph supplied

        Args:
            mol (Species): The species
        """
        # no dihedrals possible with less than 4 atoms
        if mol.n_atoms < 4:
            return

        assert mol.graph is not None

        for o, p in list(mol.graph.edges):
            for m in mol.graph.neighbors(o):
                if m == p:
                    continue

                for n in mol.graph.neighbors(p):
                    if n == o:
                        continue

                    # avoid triangle rings like cyclopropane
                    if n == m:
                        continue

                    if _is_dihedral_well_defined(mol, m, o, p, n):
                        self.add(PrimitiveDihedralAngle(m, o, p, n))

        return None

    @staticmethod
    def _get_linear_chains(mol: "Species") -> List[List[int]]:
        """
        Obtain a list of all the continuous chains of linear atoms
        present in the species i.e. A--B--C--D--E...

        Args:
            mol: The species

        Returns:
            (list[list[int]): A list of lists, each containing indices
                            of the atoms of linear chains in order
        """
        assert mol.graph is not None

        def extend_chain(chain: List[int]):
            """Extend a chain in-place"""
            for idx in range(mol.n_atoms):
                if idx in chain:
                    continue

                if mol.angle(chain[1], chain[0], idx) > _lin_thresh:
                    chain.insert(0, idx)
                    continue

                if mol.angle(chain[-2], chain[-1], idx) > _lin_thresh:
                    chain.append(idx)
                    continue
            return None

        linear_chains: List[list] = []
        for b in range(mol.n_atoms):
            for a, c in itertools.combinations(mol.graph.neighbors(b), r=2):
                if any(
                    a in chain and b in chain and c in chain
                    for chain in linear_chains
                ):
                    continue
                if mol.angle(a, b, c) > _lin_thresh:
                    chain = [a, b, c]
                    extend_chain(chain)
                    linear_chains.append(chain)

        return linear_chains

    def _add_chain_dihedrals_from_species(self, mol: "Species"):
        """
        Add extra dihedrals for chain molecules like allene, which
        are required to cover all degrees of freedom

        Args:
            mol:

        Returns:

        """
        assert mol.graph is not None
        linear_chains = self._get_linear_chains(mol)

        for chain in linear_chains:
            o, p = chain[0], chain[-1]
            for m in mol.graph.neighbors(o):
                if m == p:
                    continue

                if m in chain:
                    continue

                for n in mol.graph.neighbors(p):
                    if n == o:
                        continue

                    if n == m:
                        continue

                    if n in chain:
                        continue

                    if _is_dihedral_well_defined(mol, m, o, p, n):
                        self.add(PrimitiveDihedralAngle(m, o, p, n))

        return None


def _connect_graph_for_species(mol: "Species") -> None:
    """
    Creates a fully connected graph from the graph of a species, by
    (1) joining disconnected fragments by their shortest distance,
    (2) connecting constrained bonds, (3) joining hydrogen bonds,
    if present. The molecular graph is modified in-place.

    Args:
        mol: A species (must have atoms and graph)
    """
    assert mol.graph is not None, "Species must have graph!"

    # join hydrogen bonds
    h_bond_x = ["N", "O", "F", "P", "S", "Cl"]
    for i, j in itertools.combinations(range(mol.n_atoms), r=2):
        if (
            mol.atoms[i].label in h_bond_x
            and mol.atoms[j].label == "H"
            or mol.atoms[j].label in h_bond_x
            and mol.atoms[i].label == "H"
        ):
            vdw_sum = mol.atoms[i].vdw_radius + mol.atoms[j].vdw_radius
            if mol.distance(i, j) < 0.9 * vdw_sum:
                if not mol.graph.has_edge(i, j):
                    mol.graph.add_edge(i, j, pi=False, active=False)

    # join disconnected graph components
    if not mol.graph.is_connected:
        components = mol.graph.connected_components()
        for comp_i, comp_j in itertools.combinations(components, r=2):
            min_dist = float("inf")
            min_pair = (-1, -1)
            for i, j in itertools.product(list(comp_i), list(comp_j)):
                if mol.distance(i, j) < min_dist:
                    min_dist = mol.distance(i, j)
                    min_pair = (i, j)
            mol.graph.add_edge(*min_pair, pi=False, active=False)

    assert mol.graph.is_connected, "Unknown error in connecting graph"

    # The constraints should be counted as bonds
    if mol.constraints.distance is not None:
        for i, j in mol.constraints.distance:
            if not mol.graph.has_edge(i, j):
                mol.graph.add_edge(i, j, pi=False, active=False)

    return None


def _is_dihedral_well_defined(mol, a, b, c, d) -> bool:
    """
    A dihedral a--b--c--d is only well-defined when the
    two constituent angles are not linear

    Args:
        mol:
        a:
        b:
        c:
        d:

    Returns:
        (bool): True if well-defined otherwise False
    """
    zero_angle_thresh = np.pi - _lin_thresh
    is_linear_1 = (
        mol.angle(a, b, c) > _lin_thresh
        or mol.angle(a, b, c) < zero_angle_thresh
    )
    is_linear_2 = (
        mol.angle(b, c, d) > _lin_thresh
        or mol.angle(b, c, d) < zero_angle_thresh
    )
    return not (is_linear_1 or is_linear_2)
