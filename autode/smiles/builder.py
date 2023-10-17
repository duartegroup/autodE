import numpy as np
import networkx as nx
from autode.smiles import atom_types
from autode.log import logger
from autode.utils import log_time
from autode.atoms import Atom, AtomCollection
from autode.mol_graphs import MolecularGraph
from autode.smiles.base import SMILESAtom, SMILESBond, SMILESStereoChem
from autode.smiles.angles import SDihedral, SDihedrals, SAngle, SAngles
from ade_dihedrals import rotate, closed_ring_coords
from ade_rb_opt import opt_rb_coords
from autode.exceptions import (
    SMILESBuildFailed,
    FailedToSetRotationIdxs,
    FailedToAdjustAngles,
)


class Builder(AtomCollection):
    """
    3D geometry builder::

        Atoms:  C, 4H               H  H
        Bonds:  4 x C-H      -->     C
                                   H  H

    """

    def __init__(self):
        """
        Coordinate builder initialised from a set of atoms and bonds connecting
        them. This builder should generate something *reasonable* that can
        be cleaned up with a forcefield
        """
        super().__init__()

        self.atoms = None  # list(SMILESAtom)
        self.bonds = None  # SMILESBonds
        self.graph = None  # nx.Graph
        self.rings_idxs = None  # Iterator for atom indexes in all rings

        # A queue of atom indexes, the neighbours for which need to be added
        self.queued_atoms = []

        # A queue of dihedrals that need to be applied
        self.queued_dihedrals = SDihedrals()

    @property
    def built(self):
        """Have all the atoms been shifted appropriately?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return self.atoms is not None and len(self.queued_atoms) == 0

    @property
    def canonical_atoms(self):
        """Generate canonical autodE atoms from this set

        -----------------------------------------------------------------------
        Returns:
            (list(autode.atoms.Atom)): Atoms
        """

        atoms = []
        for atom in self.atoms:
            x, y, z = atom.coord
            atoms.append(
                Atom(atom.label, x=x, y=y, z=z, atom_class=atom.atom_class)
            )

        return atoms

    @property
    def canonical_atoms_at_origin(self):
        """Canonical set of autodE atoms all located at the origin

        -----------------------------------------------------------------------
        Returns:
            (list(autode.atoms.Atom)): Atoms all with atom.coord = [0, 0, 0]
        """
        return [Atom(atom.label) for atom in self.atoms]

    @property
    def built_atom_idxs(self):
        """Atom indexes that have been built

        -----------------------------------------------------------------------
        Returns:
            (list(int)): Atom indexes
        """
        return [i for i in range(self.n_atoms) if self.atoms[i].is_shifted]

    @property
    def non_bonded_idx_matrix(self):
        """
        Generate a matrix of ones if atoms are non-bonded and zero if for
        self pairs or they are bonded

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): shape = (n_atoms, n_atoms)
        """

        idxs = np.ones(shape=(self.n_atoms, self.n_atoms), dtype="i4")
        np.fill_diagonal(idxs, 0)  # Exclude self-repulsion

        for bond in self.bonds:
            idx_i, idx_j = bond
            idxs[idx_i, idx_j] = idxs[idx_j, idx_i] = 0

        # Do not include any atoms that have yet to be built
        for i, atom in enumerate(self.atoms):
            if not atom.is_shifted:
                idxs[i, :] = idxs[:, i] = 0

        return idxs

    @property
    def max_ring_n(self):
        """Maximum ring size in this molecule

        -----------------------------------------------------------------------
        Returns:
            (int): Maximum ring size
        """

        if self.rings_idxs is None or len(self.rings_idxs) == 0:
            return 0

        return max(len(idxs) for idxs in self.rings_idxs)

    def _atom_is_d8(self, idx):
        """
        Is an atom a d8 metal? Only consider a subset of the platinum group
        elements

        -----------------------------------------------------------------------
        Arguments:
            idx (int):

        Returns:
            (bool):
        """
        atom = self.atoms[idx]

        if atom.atomic_symbol not in ["Rh", "Pd", "Ir", "Pt"]:
            return False

        dn = atom.group - atom.charge  # Initial number of d electrons

        for bond in self.bonds.involving(idx):
            # Only remove an electron if a ligand is singly bonded (X) and
            # treat all double bonds as L2 ligands, rather than X2
            if bond.order % 2 == 1:
                dn -= 1

        logger.info(f"{atom}, dn = {dn}")
        return dn == 8

    def _explicit_all_hydrogens(self):
        """Convert all implicit hydrogens to explicit ones"""

        h_atoms = []
        for idx, atom in enumerate(self.atoms):
            if not hasattr(atom, "n_hydrogens") or atom.n_hydrogens is None:
                logger.warning(
                    f"{atom} did not have a defined number of "
                    "hydrogens. Assuming 0"
                )
                atom.n_hydrogens = 0

            for _ in range(atom.n_hydrogens):
                h_atoms.append(SMILESAtom("H", n_hydrogens=0))

                # Add the bond between the current atom and the new H
                h_idx = self.n_atoms + len(h_atoms) - 1
                self.bonds.append(SMILESBond(idx, h_idx, symbol="-"))

            # zero the number of implicit hydrogens bonded to this atom now
            # they are explicit
            atom.n_hydrogens = 0

        self.atoms += h_atoms
        return

    def _set_atom_types(self):
        """
        Set the atom types for all atoms, where the atom type is determined
        by the number of bonded atoms, and the 'hybridisation' as well as
        the stereochemistry
        """
        logger.info(f"Setting {self.n_atoms} atom types")

        self.rings_idxs = nx.minimum_cycle_basis(self.graph)
        logger.info(f"Have {len(self.rings_idxs)} ring(s)")

        for i, atom in enumerate(self.atoms):
            atom.coord = np.zeros(3)
            atom.neighbours = list(self.graph.neighbors(i))
            atom.in_ring = len(self._ring_idxs([i], return_empty=True)) > 0

            if not isinstance(atom, SMILESAtom):
                raise SMILESBuildFailed("Builder requires SMILESAtom-s")

            if atom.n_bonded == 0:
                # No type is needed for an isolated atom
                continue

            elif atom.n_bonded == 1:  # e.g. H2, FCH3
                atom.type = atom_types.TerminalAtom()

            elif atom.n_bonded == 2:  # e.g. OH2, SR2
                if atom.group == 16:
                    atom.type = atom_types.BentAtom()

                elif atom.group == 15:  # e.g. H2C=NH
                    atom.type = atom_types.TrigonalAtom()

                else:  # e.g. AuR2
                    atom.type = atom_types.LinearAtom()

            elif atom.n_bonded == 3:  # e.g. NH3
                if atom.group == 15:
                    atom.type = atom_types.TrigonalPyramidalAtom()

                else:  # e.g. BH3
                    atom.type = atom_types.TrigonalAtom()

            elif atom.n_bonded == 4:  # e.g. CH4
                if atom.atomic_symbol == "Xe":  # e.g. XeF4
                    atom.type = atom_types.SquarePlanarAtom()

                # Second row transition metals that are d8 should be sq planar
                elif self._atom_is_d8(idx=i) and atom.period == 5:
                    atom.type = atom_types.SquarePlanarAtom()

                elif atom.stereochem == SMILESStereoChem.TET_NORMAL:
                    atom.type = atom_types.TetrahedralNAtom()

                elif atom.stereochem == SMILESStereoChem.TET_INVERTED:
                    atom.type = atom_types.TetrahedralIAtom()

                else:
                    atom.type = atom_types.TetrahedralAtom()

            elif atom.n_bonded == 5:
                atom.type = atom_types.TrigonalBipyramidalAtom()

            elif atom.n_bonded == 6:
                atom.type = atom_types.OctahedralAtom()

            elif atom.n_bonded == 7:
                atom.type = atom_types.PentagonalBipyramidalAtom()

            elif atom.n_bonded == 8:
                atom.type = atom_types.SquareAntiprismAtom()

            else:
                raise NotImplementedError(
                    "Coordination numbers >8 are not" "(yet) supported"
                )

        return None

    def _ring_idxs(self, inc_idxs, return_empty=False):
        """Indexes of atoms in the ring containing this bond

        -----------------------------------------------------------------------
        Arguments:
            inc_idxs (list(int)): List of atom indexes that need to be included
                                  in the ring

        Keyword Arguments:
            return_empty (bool):

        Returns:
            (list(int)): Atom indexes in this ring if they can be found

        Raises:
            (autode.exceptions.SMILESBuildFailed): If there is no such ring
        """
        try:
            return next(
                idxs
                for idxs in self.rings_idxs
                if all(idx in idxs for idx in inc_idxs)
            )

        except StopIteration:
            if return_empty:
                return []

            raise SMILESBuildFailed(f"No ring containing {inc_idxs}")

    def _ring_path(self, ring_bond):
        """
        Find the path which traverses a ring closed by a ring bond

                 C2----C3
               /       |        -->  1, 2, 3, 4
             C1  ****  C4
                 ^
             ring bond

        -----------------------------------------------------------------------
        Arguments:
            ring_bond (autode.smiles.SMILESBond):

        Returns:
            (nx.path_generator):

        Raises:
            (SMILESBuildFailed): If a suitable path is not found
        """
        ring_idxs = self._ring_idxs(ring_bond)

        paths = nx.shortest_simple_paths(
            self.graph, source=ring_bond[0], target=ring_bond[1]
        )

        for possible_path in paths:
            # Can always have a path that traverses the ring bond (C1-C4 above)
            if len(possible_path) == 2:
                continue

            # For multiple fused rings there may be other paths that could be
            # traversed, so only take the one that has the appropriate idxs
            if all(idx in ring_idxs for idx in possible_path):
                return possible_path

        raise SMILESBuildFailed("Could not find path in ring")

    def _ring_dihedrals(self, ring_bond):
        """
        Given a ring bond find all the rotatable dihedrals that can be adjusted
        to close it with a reasonable bond distance

        -----------------------------------------------------------------------
        Arguments:
            ring_bond (autode.smiles.SMILESBond):

        Yields:
            (iterator(autode.smiles.builder.Dihedral)):

        Raises:
            (autode.exceptions.SMILESBuildFailed): If dihedrals cannot be
                                                   located
        """
        path = self._ring_path(ring_bond=ring_bond)

        # The dihedrals are then all the 4 atom tuples in sequence
        dihedral_idxs = [tuple(path[i : i + 4]) for i in range(len(path) - 3)]

        # so only add the indexes where the bond (edge) order is one
        for i, dihedral_idxs in enumerate(dihedral_idxs):
            dihedral = SDihedral(dihedral_idxs)

            # Optimum distance between the two middle atoms, used for
            # determining if a bond exists thus a dihedral can be rotated
            dihedral.mid_dist = self.bonds.first_involving(
                *dihedral.mid_idxs
            ).r0

            # If both atoms either side of this one are 'pi' atoms e.g. in a
            # benzene ring, then the ideal angle must be 0 to close the ring
            if all(self.atoms[idx].is_pi() for idx in dihedral.mid_idxs):
                dihedral.phi_ideal = 0.0

            # Only yield single bonds, that can be rotated freely
            if self.graph.get_edge_data(*dihedral.mid_idxs)["order"] == 1:
                yield dihedral

    def _reset_queued_atom_sites(self, other_idxs=None):
        """
        When a dihedral rotation(s) is(are) performed the rotation is not
        applied to the empty sites that are present in the queued atoms,
        they therefore need to be reset

        -----------------------------------------------------------------------
        Keyword Arguments:
            other_idxs (list | set | None): Other indexes that need to be reset
        """
        for idx_i in set(
            self.queued_atoms
            + list(other_idxs if other_idxs is not None else [])
        ):
            logger.info(f"Resetting sites on atom {idx_i}")

            atom = self.atoms[idx_i]
            points = [self.atoms[idx].coord for idx in atom.neighbours]

            # Resetting an atom onto two atoms can fail to apply the stereochem
            # thus only set it onto one
            if atom.has_stereochem and len(points) == 2:
                points = points[:1]

            atom.type.reset_onto(points, coord=atom.coord)

        return None

    @log_time(prefix="Closed ring in:", units="ms")
    def _adjust_ring_dihedrals(self, ring_bond, dihedrals):
        """Outsource the ring closure to an external function"""
        logger.info("Adjusting ring dihedrals to close the ring")

        coords = closed_ring_coords(
            py_coords=self.coordinates,
            py_curr_angles=dihedrals.values(self.atoms),
            py_ideal_angles=dihedrals.ideal_angles,
            py_axes=dihedrals.axes,
            py_rot_idxs=dihedrals.rot_idxs,
            py_origins=dihedrals.origins,
            py_rep_idxs=self.non_bonded_idx_matrix,
            py_close_idxs=np.array(tuple(ring_bond), dtype="i4"),
            py_r0=ring_bond.r0,
        )
        self.coordinates = coords
        return

    def _adjust_ring_angles(self, ring_bond):
        """Shift angles in a ring to close e.g. in a cyclopropane the 109º
        angles between carbons are much to large to generate a sensible
        geometry no matter the dihedral angles, so compress the C-C-C angles
        to 60º to close the ring e.g::


              C2---- C3                           C2
            /                     -->           /   |
           C1                                 C1 ---C3

        -----------------------------------------------------------------------
        Arguments:
            ring_bond (autode.smiles.base.RingBond):
        """

        path = self._ring_path(ring_bond=ring_bond)
        ring_n = len(path)

        if ring_n >= 5:
            logger.warning("Closing large rings not implemented")
            raise FailedToAdjustAngles

        angles_idxs = [tuple(path[i : i + 3]) for i in range(len(path) - 2)]
        logger.info(f"Adjusting {len(angles_idxs)} angles to close a ring")

        angles = SAngles()

        for angle_idxs in angles_idxs:
            graph = self.graph.copy()
            graph.remove_edge(ring_bond[0], ring_bond[1])

            angle = SAngle(
                idxs=angle_idxs, phi0=(np.pi - (2.0 * np.pi / ring_n))
            )

            try:
                angle.find_rot_idxs(graph=graph, atoms=self.atoms)

            except FailedToSetRotationIdxs:
                logger.warning(f"Could not adjust angle {angle_idxs}")
                raise FailedToAdjustAngles

            angle_alt = SAngle(
                idxs=angle_idxs,
                rot_idxs=angle.inverse_rot_idxs(self.atoms),
                phi0=angle.phi0,
            )

            angles.append(angle)
            angles.append(angle_alt)

        coords = self.coordinates
        axes = []
        rot_idxs = [angle.rot_idxs for angle in angles]

        for i, angle in enumerate(angles):
            idx_x, idx_y, idx_z = angle.idxs

            axis = np.cross(
                coords[idx_x, :] - coords[idx_y, :],
                coords[idx_z, :] - coords[idx_y, :],
            )

            # Alternate between forward and reverse rotations
            if i % 2 == 0:
                axis *= -1

            # Append the axis onto the coordinates
            coords = np.concatenate(
                (coords, np.expand_dims(axis + coords[idx_y, :], axis=0))
            )

            # Now the axis is coords[-1] - coods[idx_y], so
            axes.append([coords.shape[0] - 1, idx_y])

            # Append zeros to the rotated indexes corresponding to the
            # number of added coordinates (axes)
            rot_idxs[i] += len(angles) * [0]

        coords = rotate(
            py_coords=coords,
            py_angles=angles.dvalues(self.atoms) / 2,
            py_axes=np.array(axes, dtype="i4"),
            py_rot_idxs=np.array(rot_idxs, dtype="i4"),
            py_origins=angles.origins,
        )

        self.coordinates = coords[: -len(angles), :]
        return

    def _ff_distance_matrix(self, dist_consts=None):
        """Generate a distance matrix for all pairs of atoms and
        update any distance constraints that are required

        -----------------------------------------------------------------------
        Arguments:
            dist_consts (dict | None): Keyed with atoms pairs and values of
                                       the distances
        """

        dist_consts = dist_consts if dist_consts is not None else {}
        built_idxs = self.built_atom_idxs

        r0 = np.zeros((len(built_idxs), len(built_idxs)), dtype="f8")

        for bond in self.bonds:
            idx_i, idx_j = bond

            if idx_i not in built_idxs or idx_j not in built_idxs:
                continue

            # Indexes are different as only a subset of atoms will
            # be minimised and their coordinates set
            i, j = built_idxs.index(idx_i), built_idxs.index(idx_j)

            # This pair is bonded and has an already set ideal distance
            r0[i, j] = r0[j, i] = bond.r0

            if bond.order != 2:
                continue

            if bond.in_ring(self.rings_idxs) and bond.is_cis(self.atoms):
                logger.info("cis double bond in ring not adding constraint")
                continue

            logger.info("Double bond - adding constraint")
            try:
                idx_in = next(
                    idx
                    for idx in iter(self.atoms[idx_i].neighbours)
                    if self.atoms[idx].is_shifted
                )
                idx_jn = next(
                    idx
                    for idx in iter(self.atoms[idx_j].neighbours)
                    if self.atoms[idx].is_shifted
                )

                pair = (idx_in, idx_jn)

            except StopIteration:
                logger.warning(
                    "Could not fix stereochemistry, no neighbours "
                    "to add constraints to"
                )
                continue

            # A single distance constraint will be enough?!
            if all(p not in dist_consts for p in (pair, reversed(pair))):
                dist_consts[pair] = self.distance(*pair)

        # Set the items in the distance matrix, given that this may be a subset
        # of the full atoms, with different indexes
        for (idx_i, idx_j), distance in dist_consts.items():
            i, j = built_idxs.index(idx_i), built_idxs.index(idx_j)
            r0[i, j] = r0[j, i] = distance

        return r0

    def _ff_minimise(self, distance_constraints=None):
        """Minimise all built atoms using a forcefield"""

        built_idxs = self.built_atom_idxs
        n_atoms = len(built_idxs)

        # Define ideal distances for pairs of atoms that are bonded
        r0 = self._ff_distance_matrix(distance_constraints)
        bond_matrix = np.zeros(shape=(n_atoms, n_atoms), dtype=bool)
        bond_matrix[r0 != 0.0] = True

        # No repulsion between bonded atoms
        c = np.ones((n_atoms, n_atoms), dtype="f8")
        c -= np.asarray(bond_matrix, dtype="f8")
        c *= 0.8

        # and less repulsion between H and other atoms
        h_idxs = np.array(
            [
                built_idxs.index(idx)
                for idx in built_idxs
                if self.atoms[idx].label == "H"
            ],
            dtype=int,
        )
        c[h_idxs, h_idxs] *= 0.01

        # Now minimise all coordinates that are bonded
        coords = self.coordinates
        opt_cs = opt_rb_coords(
            py_coords=coords[built_idxs],
            py_bonded_matrix=bond_matrix,
            py_r0_matrix=np.asarray(r0, dtype="f8"),
            py_k_matrix=np.ones((n_atoms, n_atoms), dtype="f8"),
            py_c_matrix=c,
            py_exponent=4,
        )

        # Set the partial coordinate set
        coords[built_idxs] = opt_cs
        self.coordinates = coords
        return None

    def _close_ring(self, ring_bond):
        """
        Adjust ring dihedrals such that a ring is formed

        -----------------------------------------------------------------------
        Arguments:
            ring_bond (autode.smiles.SMILESBond):
        """
        logger.info(f"Closing ring on: {ring_bond} and adjusting atoms")

        dihedrals = SDihedrals()
        for dihedral in self._ring_dihedrals(ring_bond):
            # Generate a graph without the ring or this dihedral to locate
            # the indexes that should be rotated
            graph = self.graph.copy()
            graph.remove_edge(*ring_bond)

            try:
                dihedral.find_rot_idxs(graph=graph, atoms=self.atoms)

            except FailedToSetRotationIdxs:
                logger.warning(
                    f"Could not rotate dihedral {dihedral} "
                    f"splitting across {dihedral.mid_idxs} did not "
                    f"afford two fragments"
                )
                continue

            dihedrals.append(dihedral)

        if len(dihedrals) == 0:
            logger.info("No dihedrals to adjust to close the ring")

        else:
            self._adjust_ring_dihedrals(ring_bond, dihedrals=dihedrals)

        if not np.isclose(
            ring_bond.distance(self.atoms), ring_bond.r0, atol=0.2
        ):
            logger.info(f"A ring was poorly closed - adjusting angles")

            try:
                self._adjust_ring_angles(ring_bond)

            except FailedToAdjustAngles:
                logger.warning(
                    "Failed to close a ring, minimising on " "all atoms"
                )
                self._ff_minimise()

        self._reset_queued_atom_sites(other_idxs=ring_bond)
        return None

    @log_time(prefix="Performed final dihedral rotation in:", units="ms")
    def _minimise_non_ring_dihedrals(self):
        """
        Minimise the repulsive pairwise energy with respect to all non-ring
        dihedral rotations::

                      Z
                      |
              X -----Y
             /
            W

        """
        logger.info("Minimising non-bonded repulsion by dihedral rotation")

        dihedrals = SDihedrals()

        for bond in self.bonds:
            if bond.order != 1:
                continue

            # Check that both atoms that form this bond have > 1 neighbours,
            # thus define a dihedral
            idx_x, idx_y = bond

            # Find the other atoms that form the 4 atom tuple
            try:
                idx_w = next(
                    idx
                    for idx in self.atoms[idx_x].neighbours
                    if idx != idx_y and self.atoms[idx].n_bonded > 1
                )
                idx_z = next(
                    idx
                    for idx in self.atoms[idx_y].neighbours
                    if idx != idx_x and self.atoms[idx].n_bonded > 1
                )

            except StopIteration:
                continue  # No suitable neighbours

            dihedral = SDihedral(idxs=[idx_w, idx_x, idx_y, idx_z])

            try:
                dihedral.find_rot_idxs(self.graph.copy(), atoms=self.atoms)

            except FailedToSetRotationIdxs:
                continue  # Bond could be in a ring etc.

            dihedrals.append(dihedral)

        if len(dihedrals) == 0:
            return  # No rotation required

        logger.info(f"Have {len(dihedrals)} dihedrals to rotate")

        coords = rotate(
            py_coords=self.coordinates,
            py_angles=np.zeros(len(dihedrals)),
            py_axes=dihedrals.axes,
            py_rot_idxs=dihedrals.rot_idxs,
            py_origins=dihedrals.origins,
            minimise=True,
            py_rep_idxs=self.non_bonded_idx_matrix,
        )

        self.coordinates = coords
        return None

    def _force_double_bond_stereochem(self, dihedral):
        """
        For double bonds in rings (>8 members usually) stereochemistry needs to
        be generated, but may not be possible, so minimise the energy under
        the constraint defining the E/Z over a specific dihedral

                      Z
                      |
              X -----Y
             /
            W

        -----------------------------------------------------------------------
        Arguments:
            dihedral (autode.smiles.builder.Dihedral):
        """
        logger.info(f"Forcing stereochemistry for {dihedral}")

        if not (
            self.graph.edges[dihedral.mid_idxs]["order"] == 2
            and np.isclose(dihedral.phi0 % np.pi, 0)
        ):
            raise ValueError(
                "Expecting a 0º or 180º dihedral for E/Z"
                "over a double bond - cannot rotate"
            )

        # Get the bond lengths for the three bonds
        r_wx = self.bonds.first_involving(*dihedral.idxs[:2]).r0
        r_xy = self.bonds.first_involving(*dihedral.mid_idxs).r0
        r_yz = self.bonds.first_involving(*dihedral.idxs[-2:]).r0

        if np.isclose(dihedral.phi0, np.pi):
            # Distance constraint for a trans double bond
            r_wz = np.sqrt(
                ((r_wx + r_yz) * np.sin(np.pi / 3.0)) ** 2
                + ((r_wx + r_yz) * np.cos(np.pi / 3.0) + r_xy) ** 2
            )

        else:  # and similarly for cis
            r_wz = (r_wx + r_yz) * np.sin(np.pi / 6.0) + r_xy

        def c_cosine_rule(a, b, gamma):
            """c = √a^2 + b^2 - 2ab cos(γ)"""
            return np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(gamma))

        r_wy = c_cosine_rule(r_wx, r_xy, 2.0 * np.pi / 3.0)
        r_xz = c_cosine_rule(r_xy, r_yz, 2.0 * np.pi / 3.0)

        # Apply distance constraints over the all the pairwise distances,
        # such that the correct geometry is the only minimum (with just r_wz)
        # constraints the WXY and XYZ angles change to accommodate r_wz, rather
        # than there being any dihedral rotation)
        dist_consts = {
            dihedral.end_idxs: r_wz,
            (dihedral.idxs[0], dihedral.idxs[2]): r_wy,
            (dihedral.idxs[1], dihedral.idxs[3]): r_xz,
            dihedral.mid_idxs: r_xy,
        }

        self._ff_minimise(distance_constraints=dist_consts)
        self._reset_queued_atom_sites(other_idxs=dihedral.mid_idxs)
        return None

    def _queue_double_bond_dihedral(self, bond):
        """
        For a double bond queue the dihedral rotation to be applied such that::

              X -----Y
             /       |
            W        Z

        where the dihedral is 0 or π, depending on the stereochemistry

        -----------------------------------------------------------------------
        Arguments:
            bond (autode.smiles.base.SMILESBond):
        """
        idx_x, idx_y = bond

        nbrs_x = [idx for idx in self.atoms[idx_x].neighbours if idx != idx_y]
        nbrs_y = [idx for idx in self.atoms[idx_y].neighbours if idx != idx_x]

        if len(nbrs_x) == 0 or len(nbrs_y) == 0:
            logger.info(
                f"At least one atom forming {bond} had no "
                "neighbours - no need to rotate the dihedral"
            )
            return

        # Remove any hydrogen atoms from the neighbours, as they are skipped
        # when defining the stereochem
        nbrs_x_noH = [idx for idx in nbrs_x if self.atoms[idx].label != "H"]
        nbrs_y_noH = [idx for idx in nbrs_y if self.atoms[idx].label != "H"]

        if len(nbrs_x_noH) > 0:
            nbrs_x = nbrs_x_noH

        if len(nbrs_y_noH) > 0:
            nbrs_y = nbrs_y_noH

        # Index W is the closest atom index to X, that isn't Y
        idx_w = nbrs_x[np.abs(np.array(nbrs_x) - idx_x).argmin()]
        # and similarly for Z
        idx_z = nbrs_y[np.abs(np.array(nbrs_y) - idx_y).argmin()]

        # Is this bond cis or trans?
        stro_x, stro_y = (
            self.atoms[idx_x].stereochem,
            self.atoms[idx_y].stereochem,
        )

        phi = np.pi  # Default to a trans double bond

        if (
            (
                all(
                    self.atoms[idx].in_ring
                    for idx in (idx_w, idx_x, idx_y, idx_z)
                )
                and not self.atoms[idx_x].has_stereochem
            )
            or stro_x == stro_y == SMILESStereoChem.ALKENE_UP
            or stro_x == stro_y == SMILESStereoChem.ALKENE_DOWN
        ):
            phi = 0

        dihedral = SDihedral([idx_w, idx_x, idx_y, idx_z], phi0=phi)

        logger.info(f"Queuing {dihedral}")
        self.queued_dihedrals.append(dihedral)
        return None

    def _rotate_dihedrals(self):
        """Rotate all dihedrals in the queue"""
        if len(self.queued_dihedrals) == 0:
            return  # Nothing to be done

        logger.info(f"Have {len(self.queued_dihedrals)} dihedral(s) to rotate")

        for i, dihedral in enumerate(self.queued_dihedrals):
            try:
                dihedral.find_rot_idxs(
                    graph=self.graph.copy(), atoms=self.atoms
                )

            except FailedToSetRotationIdxs:
                logger.warning(f"Could not apply rotation {dihedral}")

                if dihedral.needs_forcing(atoms=self.atoms):
                    logger.info(
                        "Dihedral is too far away from that defined "
                        "by the stereochemistry - forcing"
                    )
                    self._force_double_bond_stereochem(dihedral)

                # Delete this dihedral, that has beed forced, and continue
                del self.queued_dihedrals[i]
                return self._rotate_dihedrals()

        dphis = [
            dihedral.phi0 - dihedral.value(self.atoms)
            for dihedral in self.queued_dihedrals
        ]

        self.coordinates = rotate(
            py_coords=self.coordinates,
            py_angles=np.array(dphis, dtype="f8"),
            py_axes=self.queued_dihedrals.axes,
            py_rot_idxs=self.queued_dihedrals.rot_idxs,
            py_origins=self.queued_dihedrals.origins,
        )

        self.queued_dihedrals.clear()
        self._reset_queued_atom_sites()
        return None

    def _add_bonded_atoms(self, idx):
        """
        Add all the atoms bonded to a particular index, that have not already
        been shifted

        -----------------------------------------------------------------------
        Arguments:
            idx (int): Atom index
        """
        atom = self.atoms[idx]

        for bond in self.bonds.involving(idx):
            bonded_idx = bond[0] if bond[1] == idx else bond[1]

            if bonded_idx in self.queued_atoms:
                # Delete one of the empty sites
                if atom.type.n_empty_sites > 0:
                    _ = atom.type.empty_site()

                self._close_ring(ring_bond=bond)
                continue

            if self.atoms[bonded_idx].is_shifted:
                # Dihedrals over double bonds need to be 0 or π, queue the
                # rotation to be performed after all other atoms have been
                # added
                if bond.order == 2:
                    self._queue_double_bond_dihedral(bond)

                continue

            # Get an empty site on this atom. If this atom is chrial then
            # there is no choice to minimise the repulsion with the rest of
            # the structure
            if atom.type.is_chiral:
                site = atom.type.empty_site()
            else:
                site = atom.type.empty_site_mr(
                    atom.coord, other_coords=self.coordinates
                )

            # Coordinate of this atom is the current position shifted by
            # the ideal distance in a direction of a empty coordination
            # site on the atom
            coord = bond.r0 * site + atom.coord
            bonded_atom = self.atoms[bonded_idx]
            bonded_atom.translate(coord)

            # Atoms that are not terminal need to be added to the queue
            if not isinstance(
                self.atoms[bonded_idx].type, atom_types.TerminalAtom
            ):
                # and the atom type rotated so an empty site is coincident
                # with this atom
                bonded_atom.type.rotate_empty_onto(
                    point=atom.coord, coord=bonded_atom.coord
                )
                # and queue
                self.queued_atoms.append(bonded_idx)

        return None

    def set_atoms_bonds(self, atoms, bonds):
        """
        From a list of SMILESAtoms, and SMILESBonds set the required attributes
        and convert all implicit hydrogens into explicit atoms

        -----------------------------------------------------------------------
        Arguments:
            atoms (list(autode.smiles.base.SMILESAtom)):

            bonds (auode.smiles.base.SMILESBonds):
        """
        if atoms is None or len(atoms) == 0:
            raise SMILESBuildFailed("Cannot build a structure with no atoms")

        # Set attributes
        self.atoms, self.bonds = atoms, bonds
        self.graph = MolecularGraph()
        self.queued_atoms = []
        self.queued_dihedrals = SDihedrals()

        self._explicit_all_hydrogens()

        # Add nodes for all the atom indexes, without attributes for e.g
        # atomic symbol as a normal molecular graph would have
        for i in range(self.n_atoms):
            self.graph.add_node(i)

        # Set the ideal bond lengths and the graph edges
        for bond in self.bonds:
            idx_i, idx_j = bond
            self.graph.add_edge(idx_i, idx_j, order=bond.order)

            bond.r0 = self.atoms.eqm_bond_distance(idx_i, idx_j)

        self._set_atom_types()

        # Add the first atom to the queue of atoms to be translated etc.
        self.queued_atoms.append(0)
        # perturb the first atom's coordinate slightly, such that it is treated
        # as being shifted (built)
        self.atoms[0].translate(vec=np.array([0.001, 0.001, 0.001]))
        return None

    @log_time(prefix="Built 3D in:", units="ms")
    def build(self, atoms, bonds):
        """
        Build a molecule by iterating through all the atoms adding it and
        each of it's neighbours. i.e.

        atoms = [C, H, H, H]

        1. Add C at origin
        2. Add all neighbours
        3. Done

        atoms = [C, C, C, 8xH]

        1. Add C at origin
        2. Add H3, C neighbours & update queued atoms to include the C
           that has been translated but needs it's neighbours adding to it

        ----------------------------------------------------------------------
        Arguments:
            atoms (list(autode.smiles.SMILESAtoms)):

            bonds (autode.smiles.SMILESBonds):
        """
        self.set_atoms_bonds(atoms, bonds)

        while not self.built:
            idx = self.queued_atoms.pop(0)
            self._add_bonded_atoms(idx)
            self._rotate_dihedrals()

            logger.info(f"Queue: {self.queued_atoms}")

        self._minimise_non_ring_dihedrals()
        return None
