"""
Internal coordinates. Notation follows:


x : Cartesian coordinates
B : Wilson B matrix
q : Primitive internal coordinates
G : Spectroscopic G matrix

Set-up of redundant primitives is based on J. Chem. Phys., 117, 2002, 9160
"""
import copy

import numpy as np
from enum import Enum
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
    from autode.mol_graphs import MolecularGraph
    from autode.opt.coordinates.cartesian import CartesianCoordinates
    from autode.opt.coordinates.primitives import (
        ConstrainedPrimitive,
        _DistanceFunction,
    )


class InternalCoordinates(OptCoordinates, ABC):  # lgtm [py/missing-equals]
    def __new__(cls, input_array) -> "InternalCoordinates":
        """New instance of these internal coordinates"""

        arr = super().__new__(cls, input_array, units="Å")

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

    def append(self, item: Primitive) -> None:
        """Append an item to this set of primitives"""
        assert isinstance(item, Primitive), "Must be a Primitive type!"
        super().append(item)

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
                self.append(self._primitive_type(i, j))

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


def build_pic_from_species(
    mol: "Species",
    aux_bonds=False,
) -> AnyPIC:
    """
    Build a set of primitives from the species, using the graph as
    a starting point for the connectivity of the species. Also joins
    any disjoint parts of the graph, and adds hydrogen bonds to
    ensure that the primitives are redundant

    Args:
        mol:
        aux_bonds:

    Returns:
        (AnyPIC): The set of primitive internals
    """
    pic = AnyPIC()
    core_graph = _get_connected_graph_from_species(mol)
    _add_bonds_from_species(pic, mol, core_graph, aux_bonds=aux_bonds)
    _add_angles_from_species(pic, mol, core_graph)
    _add_dihedrals_from_species(pic, mol, core_graph)
    return pic


def _get_connected_graph_from_species(mol: "Species") -> "MolecularGraph":
    """
    Creates a fully connected graph from a species, by (1) joining
    disconnected fragments by their shortest distance, (2) connecting
    constrained bonds, (3) joining hydrogen bonds, if present.

    Args:
        mol: A species containing atoms and coordinates

    Returns:
        (MolecularGraph):
    """
    # if graph does not exist, create one
    if mol.graph is None:
        mol.reset_graph()

    assert mol.graph is not None
    core_graph = copy.deepcopy(mol.graph)

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
                if not core_graph.has_edge(i, j):
                    core_graph.add_edge(i, j, pi=False, active=False)

    # join disconnected graph components
    if not core_graph.is_connected:
        components = core_graph.get_components()
        for comp_i, comp_j in itertools.combinations(components, r=2):
            min_dist = float("inf")
            min_pair = (-1, -1)
            for i, j in itertools.product(list(comp_i), list(comp_j)):
                if mol.distance(i, j) < min_dist:
                    min_dist = mol.distance(i, j)
                    min_pair = (i, j)
            # avoid connecting distant components
            if min_dist < 5.0:
                core_graph.add_edge(*min_pair, pi=False, active=False)

    if not core_graph.is_connected:
        raise RuntimeError(
            "Unable to join all the fragments, distance between "
            "one or more pairs of fragments is too high (>5.0 Å)"
        )

    # The constraints should be counted as bonds
    if mol.constraints.distance is not None:
        for i, j in mol.constraints.distance:
            if not core_graph.has_edge(i, j):
                core_graph.add_edge(i, j, pi=False, active=False)

    return core_graph


def _add_bonds_from_species(
    pic: AnyPIC,
    mol: "Species",
    core_graph: "MolecularGraph",
    aux_bonds: bool = False,
):
    """
    Modify the supplied AnyPIC instance in-place by adding bonds, from the
    connectivity graph supplied

    Args:
        pic: The AnyPIC instance (modified in-place)
        mol: The species object
        core_graph: The connectivity graph
        aux_bonds: Whether to add auxiliary bonds (< 2.5 * covalent radii sum)
    """
    n = 0
    for i, j in sorted(core_graph.edges):
        if (
            mol.constraints.distance is not None
            and (i, j) in mol.constraints.distance
        ):
            r = mol.constraints.distance[(i, j)]
            pic.append(ConstrainedPrimitiveDistance(i, j, r))
            n += 1
        else:
            pic.append(PrimitiveDistance(i, j))
    assert n == mol.constraints.n_distance

    if not aux_bonds:
        return None

    # add auxiliary bonds if specified
    for i, j in itertools.combinations(range(mol.n_atoms), r=2):
        if core_graph.has_edge(i, j):
            continue
        if mol.distance(i, j) < 2.5 * mol.eqm_bond_distance(i, j):
            pic.append(PrimitiveDistance(i, j))
    return None


def _add_angles_from_species(
    pic: AnyPIC,
    mol: "Species",
    core_graph: "MolecularGraph",
    lin_thresh=Angle(170, "deg"),
) -> None:
    """
    Modify the set of primitives in-place by adding angles, from the
    connectivity graph supplied

    Args:
        pic: The AnyPIC instance (modified in-place)
        mol: The species object
        core_graph: The connectivity graph
    """

    def get_ref_atom(a, b, c):
        """get a reference atom for a-b-c linear angle"""
        # all atoms in 4 A radius except a, b, c
        near_atoms = [
            idx
            for idx in range(mol.n_atoms)
            if mol.distance(b, idx) < Distance(4.0, "ang")
            and idx not in (a, b, c)
        ]
        # get atoms closest to perpendicular
        deviations_from_90 = []
        for atom in near_atoms:
            i_b_a = mol.angle(atom, b, a)
            if i_b_a > lin_thresh or i_b_a < 180 - lin_thresh:
                continue
            i_b_c = mol.angle(atom, b, c)
            if i_b_c > lin_thresh or i_b_c < 180 - lin_thresh:
                continue
            deviation_a = abs(i_b_a - Angle(90, "deg"))
            deviation_c = abs(i_b_c - Angle(90, "deg"))
            avg_dev = (deviation_a + deviation_c) / 2
            deviations_from_90.append(avg_dev)

        if len(deviations_from_90) == 0:
            return None

        return near_atoms[np.argmin(deviations_from_90)]

    for o in range(mol.n_atoms):
        for n, m in itertools.combinations(core_graph.neighbors(o), r=2):
            # avoid almost linear angles
            if mol.angle(m, o, n) < lin_thresh:
                pic.append(PrimitiveBondAngle(o=o, m=m, n=n))
            else:
                # if central atom is connected to another, no need to include
                other_neighbours = list(core_graph.neighbors(o))
                other_neighbours.remove(m)
                other_neighbours.remove(n)
                if any(
                    mol.angle(m, o, x) < lin_thresh for x in other_neighbours
                ) or any(
                    mol.angle(n, o, x) < lin_thresh for x in other_neighbours
                ):
                    continue

                # for linear bends, ideally a reference atom is needed
                r = get_ref_atom(m, o, n)
                if r is not None:
                    pic.append(
                        PrimitiveLinearAngle(m, o, n, r, LinearBendType.BEND)
                    )
                    pic.append(
                        PrimitiveLinearAngle(
                            m, o, n, r, LinearBendType.COMPLEMENT
                        )
                    )
                else:  # these use dummy atom for reference
                    pic.append(
                        PrimitiveDummyLinearAngle(m, o, n, LinearBendType.BEND)
                    )
                    pic.append(
                        PrimitiveDummyLinearAngle(
                            m, o, n, LinearBendType.COMPLEMENT
                        )
                    )

    return None


def _add_dihedrals_from_species(
    pic: AnyPIC,
    mol: "Species",
    core_graph: "MolecularGraph",
    lin_thresh=Angle(170, "deg"),
) -> None:
    """
    Modify the set of primitives in-place by adding dihedrals (torsions),
    from the connectivity graph supplied

    Args:
        pic: The AnyPIC instance (modified in-place)
        mol: The species
        core_graph: The connectivity graph
    """
    # no dihedrals possible with less than 4 atoms
    if mol.n_atoms < 4:
        return

    # find all linear atom chains A--B--C--D... and add dihedrals to terminal atoms
    linear_chains = []
    for b in range(mol.n_atoms):
        for a, c in itertools.combinations(core_graph.neighbors(b), r=2):
            if mol.angle(a, b, c) > lin_thresh:
                linear_chains.append((a, b, c))

    def concatenate_adjacent_chains(chains_list):
        for chain1, chain2 in itertools.combinations(chains_list, r=2):
            if chain1[0] == chain2[0]:
                new_chain = list(reversed(chain2)) + list(chain1)
            elif chain1[0] == chain2[-1]:
                new_chain = list(chain2) + list(chain1)
            elif chain1[-1] == chain2[0]:
                new_chain = list(chain1) + list(chain2)
            elif chain1[-1] == chain2[-1]:
                new_chain = list(chain1) + list(reversed(chain2))
            else:
                continue

            chains_list.remove(chain1)
            chains_list.remove(chain2)
            chains_list.append(tuple(new_chain))
            return concatenate_adjacent_chains(chains_list)

    concatenate_adjacent_chains(linear_chains)
    terminal_points = [(chain[0], chain[-1]) for chain in linear_chains]

    # add normal + linear chain dihedrals
    for o, p in list(core_graph.edges) + terminal_points:
        for m in core_graph.neighbors(o):
            if m == p:
                continue

            for n in core_graph.neighbors(p):
                if n == o:
                    continue

                # avoid triangle rings like cyclopropane
                if n == m:
                    continue

                is_linear_1 = mol.angle(m, o, p) > lin_thresh
                is_linear_2 = mol.angle(o, p, n) > lin_thresh

                # if any angle is linear, don't add dihedral
                if is_linear_1 or is_linear_2:
                    continue
                else:
                    pic.append(PrimitiveDihedralAngle(m, o, p, n))
    return None
