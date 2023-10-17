import numpy as np
import networkx as nx
from autode.log import logger
from autode.atoms import AtomCollection
from autode.exceptions import FailedToSetRotationIdxs, SMILESBuildFailed


class SAngle:
    """Angle used in 3D construction from SMILES"""

    def __init__(self, idxs, rot_idxs=None, phi0=None):
        """Angle between a set of atoms. In order"""

        self.idxs = idxs
        self.phi_ideal = phi0
        self.rot_idxs = rot_idxs

    def __str__(self):
        return f"Angle(idxs={self.idxs})"

    def __repr__(self):
        return self.__str__()

    def value(self, atoms) -> float:
        """

        -----------------------------------------------------------------------
        Arguments:
            atoms (list(autode.atoms.Atom)):

        Returns:
            (float): Angle in radians
        """

        idx_x, idx_y, idx_z = self.idxs
        vec1 = atoms[idx_x].coord - atoms[idx_y].coord
        vec2 = atoms[idx_z].coord - atoms[idx_y].coord

        return np.arccos(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )

    def _find_rot_idxs_from_pair(
        self, graph, atoms, pair, max_bond_distance=4.0
    ):
        """
        Split the graph across a pair of indexes and set the atom indexes
        to be rotated

        Arguments:
            graph (nx.Graph):
            atoms (list(autode.atoms.Atom)):
            pair (list(int)): len == 2

        Keyword Arguments:
            max_bond_distance (float): Maximum distance in Å that two atoms
                                       that appear in the graph edges (bonds)
                                       that constitutes a bond
        """
        graph.remove_edge(*pair)

        # Remove all the nodes in the graph that have not been shifted, thus
        # the rotation indexes only include atoms that have been 'built'
        for idx, atom in enumerate(atoms):
            if hasattr(atom, "is_shifted") and not atom.is_shifted:
                graph.remove_node(idx)

        # Delete edges that are too far away (i.e. unclosed rings)
        for idx_i, idx_j in graph.edges:
            if {idx_i, idx_j} == set(pair):
                logger.error("Cannot cut across a ring")
                continue

            if (
                np.linalg.norm(atoms[idx_i].coord - atoms[idx_j].coord)
                > max_bond_distance
            ):
                logger.info(
                    f"Bond {idx_i}-{idx_j} was not present, may "
                    f"remove from graph for idx location"
                )

                graph.remove_edge(idx_i, idx_j)

                # Removing edges is only possible if the graph remains intact,
                # i.e. there are no stranded atoms formed by splitting, sso
                # re-add the edge if two separate graphs are formed
                if not nx.is_connected(graph):
                    graph.add_edge(idx_i, idx_j)

        components = [
            graph.subgraph(c) for c in nx.connected_components(graph)
        ]

        if len(components) != 2:
            raise FailedToSetRotationIdxs(
                f"Splitting over {pair} did " "not afford two fragments"
            )

        # Choose the components that will be rotated
        cpnt_idx = 0 if pair[0] in components[0].nodes else 1

        self.rot_idxs = [
            1 if i in components[cpnt_idx].nodes else 0
            for i in range(len(atoms))
        ]
        return None

    def find_rot_idxs(self, graph, atoms):
        """Find the atom indexes to rotate by splitting rhe graph across
        the edge that appears first in the angle, e.g.::

                       Z
                      /
                X  - Y
                  ^
            split across this bond

        """
        return self._find_rot_idxs_from_pair(graph, atoms, pair=self.idxs[:2])

    def inverse_rot_idxs(self, atoms):
        """
        Return the inverse of a set of rotation indexes for e.g. rotating
        the atoms on the other side of the angle. Skip any atoms that
        have not been moved

        -----------------------------------------------------------------------
        Returns:
            (list(int)):
        """
        return [
            1
            if (hasattr(atom, "is_shifted") and atom.is_shifted)
            and self.rot_idxs[i] != 1
            else 0
            for i, atom in enumerate(atoms)
        ]

    @property
    def phi0(self):
        """A non-None ideal angle, default to 100 degrees"""
        return 1.74533 if self.phi_ideal is None else self.phi_ideal


class SAngles(list):
    @property
    def axes(self):
        raise NotImplementedError

    @property
    def origins(self):
        """Origins for the rotation, as the central atom of the trio"""
        return np.array([angle.idxs[1] for angle in self], dtype="i4")

    @property
    def rot_idxs(self):
        """Matrix of atom indexes to rotate"""
        return np.array([angle.rot_idxs for angle in self], dtype="i4")

    @property
    def ideal_angles(self):
        """Ideal angle vector (float | None)"""
        return [angle.phi_ideal for angle in self]

    def values(self, atoms):
        """Current angle vector in radians"""
        return np.array([angle.value(atoms) for angle in self], dtype="f8")

    def dvalues(self, atoms):
        """Difference between the current and ideal angles"""
        return np.array(
            [angle.phi0 - angle.value(atoms) for angle in self], dtype="f8"
        )


class SDihedrals(SAngles):
    @property
    def axes(self):
        return np.array([dihedral.mid_idxs for dihedral in self], dtype="i4")

    @property
    def origins(self):
        origins = []
        for dihedral in self:
            idx_i, idx_j = dihedral.mid_idxs
            origins.append(idx_i if dihedral.rot_idxs[idx_i] == 1 else idx_j)

        return np.array(origins, dtype="i4")


class SDihedral(SAngle):
    r"""
    A dihedral defined by 4 atom indexes used in building a 3D strucutre
    from a SMILES string e.g.::

           X       W
           |      /
           Y---- Z

    """

    def __init__(self, idxs, rot_idxs=None, phi0=None, mid_dist=2.0):
        r"""
        A dihedral constructed from atom indexes and possibly indexes that
        should be rotated, if this dihedral is altered::

             W
              \
               X --- Y
                     \
                      Z

        -----------------------------------------------------------------------
        Arguments:
            idxs (list(int)): 4 atom indexes defining the dihedral

        Keyword Arguments:
            rot_idxs (list(int) | None): Indexes to rotate, 1 if the atoms
                                         should be rotated else 0

            phi0 (float | None): Ideal angle for this dihedral (radians)

            mid_dist (float): Optimum distance between X-Y
        """
        super().__init__(idxs=idxs, rot_idxs=rot_idxs, phi0=phi0)

        # Atom indexes of the central two atoms (X, Y)
        _, idx_x, idx_y, _ = idxs

        self.mid_idxs = (idx_x, idx_y)
        self.mid_dist = mid_dist

    def __str__(self):
        return f"Dihedral(idxs={self.idxs}, φ0={round(self.phi0, 2)})"

    @property
    def end_idxs(self):
        """Atoms defining the end of the dihedral"""
        return self.idxs[0], self.idxs[-1]

    @property
    def phi0(self):
        """A non-None ideal angle for this dihedral"""
        return 0.0 if self.phi_ideal is None else self.phi_ideal

    def needs_forcing(self, atoms):
        """Does this dihedral angle need to be forced? i.e. has defined
        stereochemistry that is not respected"""

        return (
            atoms[self.mid_idxs[0]].has_stereochem
            and abs(self.dphi(atoms)) > np.pi / 3
        )

    def dphi(self, atoms):
        """∆φ = φ_curr - φ_ideal"""
        return self.value(atoms=atoms) - self.phi0

    def value(self, atoms):
        """
        Calculate the value of a dihedral defined by some atoms with non-zero
        positions

        -----------------------------------------------------------------------
        Arguments:
            atoms (list(autode.atoms.Atom)):

        Returns:
            (float): The dihedral angle in radians

        Raises:
            (SMILESBuildFailed):
        """
        atoms_ = AtomCollection(atoms)
        try:
            return float(atoms_.dihedral(*self.idxs))

        except ValueError:
            raise SMILESBuildFailed

    def find_rot_idxs(self, graph, atoms):
        """
        Find the atom indexes that should be rotated for this dihedral

        -----------------------------------------------------------------------
        Arguments:
            graph (nx.Graph):

            atoms (list(autode.atoms.Atom)):
        """
        return self._find_rot_idxs_from_pair(
            graph,
            atoms,
            pair=self.mid_idxs,
            max_bond_distance=1.5 * self.mid_dist,
        )
