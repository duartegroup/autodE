import numpy as np
import networkx as nx
from time import time
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.bonds import get_avg_bond_length
from autode.geom import get_rot_mat_kabsch
from autode.exceptions import SMILESBuildFailed, FailedToSetRotationIdxs
from autode.smiles.base import SMILESAtom, SMILESBond
from cdihedrals import rotate, closed_ring_coords


class Builder:
    """3D geometry builder

    Atoms:  C, 4H               H  H
    Bonds:  4 x C-H      -->     C
                               H  H
    """

    @property
    def built(self):
        """Have all the atoms been shifted appropriately"""
        return self.atoms is not None and len(self.queued_atoms) == 0

    @property
    def n_atoms(self):
        return 0 if self.atoms is None else len(self.atoms)

    @property
    def coordinates(self):
        """Numpy array of coordinates"""
        return np.array([a.coord for a in self.atoms], dtype='f8')

    @coordinates.setter
    def coordinates(self, value):
        """Set the coordinates from a numpy array"""
        for i, atom in enumerate(self.atoms):
            atom.coord = value[i]

    @property
    def non_bonded_idx_matrix(self):
        """
        Generate a matrix of ones if atoms are non-bonded and zero if for
        self pairs or they are bonded

        Returns:
            (np.ndarray): shape = (n_atoms, n_atoms)
        """

        idxs = np.ones(shape=(self.n_atoms, self.n_atoms), dtype='i4')
        np.fill_diagonal(idxs, 0)  # Exclude self-repulsion

        for bond in self.bonds:
            idx_i, idx_j = bond
            idxs[idx_i, idx_j] = idxs[idx_j, idx_i] = 0

        # Do not include any atoms that have yet to be built
        for i, atom in enumerate(self.atoms):
            if not atom.is_shifted:
                idxs[i, :] = idxs[:, i] = 0

        return idxs

    def _atom_is_d8(self, idx):
        """
        Is an atom a d8 metal? Only consider a subset of the platinum group
        elements

        Arguments:
            idx (int):

        Returns:
            (bool):
        """
        atom = self.atoms[idx]

        if atom.atomic_symbol not in ['Rh', 'Pd', 'Ir', 'Pt']:
            return False

        dn = atom.group - atom.charge     # Initial number of d electrons

        for bond in self.bonds.involving(idx):

            # Only remove an electron if a ligand is singly bonded (X) and
            # treat all double bonds as L2 ligands, rather than X2
            if bond.order % 2 == 1:
                dn -= 1

        logger.info(f'{atom}, dn = {dn}')
        return dn == 8

    def _explicit_all_hydrogens(self):
        """Convert all implicit hydrogens to explicit ones"""

        h_atoms = []
        for idx, atom in enumerate(self.atoms):

            if not hasattr(atom, 'n_hydrogens') or atom.n_hydrogens is None:
                raise SMILESBuildFailed('All atoms must have a defined number '
                                        f'of hydrogens to build. {atom} had '
                                        f'n_hydrogens = None')

            for _ in range(atom.n_hydrogens):
                h_atoms.append(SMILESAtom('H', n_hydrogens=0))

                # Add the bond between the current atom and the new H
                h_idx = len(self.atoms) + len(h_atoms) - 1
                self.bonds.append(SMILESBond(idx, h_idx, symbol='-'))

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
        logger.info(f'Setting {self.n_atoms} atom types')

        self.rings_idxs = nx.cycle_basis(self.graph)
        logger.info(f'Have {len(self.rings_idxs)} ring(s)')

        for i, atom in enumerate(self.atoms):

            atom.coord = np.zeros(3)
            atom.neighbours = list(self.graph.neighbors(i))
            atom.in_ring = len(self._ring_idxs([i], return_empty=True)) > 0

            # To build linear molecules the sites on atom types (templates)
            # need to be reversed for every other atom, otherwise the next atom
            # will be added to the same site forming a ring
            if atom.n_bonded == 1:                             # e.g. H2, FCH3
                atom.type = TerminalAtom()

            elif atom.n_bonded == 2:                           # e.g. OH2, SR2
                if atom.group == 16:
                    atom.type = BentAtom()

                else:                                          # e.g. AuR2
                    atom.type = LinearAtom()

            elif atom.n_bonded == 3:                           # e.g. NH3
                if atom.group == 15:
                    atom.type = TrigonalPyramidalAtom()

                else:                                          # e.g. BH3
                    atom.type = TrigonalAtom()

            elif atom.n_bonded == 4:                           # e.g. CH4

                if atom.atomic_symbol == 'Xe':                 # e.g. XeF4
                    atom.type = SquarePlanarAtom()

                # Second row transition metals that are d8 should be sq planar
                elif self._atom_is_d8(idx=i) and atom.period == 5:
                    atom.type = SquarePlanarAtom()

                elif atom.stereochem == '@':
                    atom.type = TetrahedralNAtom()

                elif atom.stereochem == '@@':
                    atom.type = TetrahedralIAtom()

                else:
                    atom.type = TetrahedralAtom()

            elif atom.n_bonded == 5:
                atom.type = TrigonalBipyramidalAtom()

            elif atom.n_bonded == 6:
                atom.type = OctahedralAtom()

            elif atom.n_bonded == 7:
                atom.type = PentagonalBipyramidalAtom()

            elif atom.n_bonded == 8:
                atom.type = SquareAntiprismAtom()

            else:
                raise NotImplementedError('Coordination numbers >8 are not'
                                          '(yet) supported')

        return None

    def _ring_idxs(self, inc_idxs, return_empty=False):
        """Indexes of atoms in the ring containing this bond

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
            return next(idxs for idxs in self.rings_idxs
                        if all(idx in idxs for idx in inc_idxs))

        except StopIteration:
            if return_empty:
                return []

            raise SMILESBuildFailed(f'No ring containing {inc_idxs}')

    def _ring_path(self, ring_bond):
        """
        Find the path which traverses a ring closed by a ring bond

                 C2----C3
               /       |        -->  1, 2, 3, 4
             C1  ****  C4
                 ^
             ring bond

        Args:
            ring_bond (autode.smiles.SMILESBond):

        Returns:
            (nx.path_generator):

        Raises:
            (SMILESBuildFailed): If a suitable path is not found
        """
        ring_idxs = self._ring_idxs(ring_bond)

        paths = nx.shortest_simple_paths(self.graph,
                                         source=ring_bond[0],
                                         target=ring_bond[1])

        for possible_path in paths:

            # Can always have a path that traverses the ring bond (C1-C4 above)
            if len(possible_path) == 2:
                continue

            # For multiple fused rings there may be other paths that could be
            # traversed, so only take the one that has the appropriate idxs
            if all(idx in ring_idxs for idx in possible_path):
                return possible_path

        raise SMILESBuildFailed('Could not find path in ring')

    def _ring_dihedrals(self, ring_bond):
        """
        Given a ring bond find all the rotatable dihedrals that can be adjusted
        to close it with a reasonable bond distance

        Arguments:
            ring_bond (autode.smiles.SMILESBond):

        Yields:
            (iterator(autode.smiles.builder.Dihedral)):

        Raises:
            (autode.exceptions.SMILESBuildFailed): If dihedrals cannot be
                                                   located
        """
        path = self._ring_path(ring_bond=ring_bond)

        # The dihedrals are then the all the 4 atom tuples in sequence
        dihedral_idxs = [tuple(path[i:i + 4]) for i in range(len(path) - 3)]

        # so only add the indexes where the bond (edge) order is one
        for i, dihedral_idxs in enumerate(dihedral_idxs):

            dihedral = Dihedral(dihedral_idxs)

            # If both atoms either side of this one are 'pi' atoms e.g. in a
            # benzene ring, then the ideal angle must be 0 to close the ring
            if all(self.atoms[idx].is_pi for idx in dihedral.mid_idxs):
                dihedral.phi_ideal = 0.0

            # Only yield single bonds, that can be rotated freely
            if self.graph.get_edge_data(*dihedral.mid_idxs)['order'] == 1:
                yield dihedral

    def _reset_queued_atom_sites(self, other_idxs=None):
        """
        When a dihedral rotation(s) is(are) performed the rotation is not
        applied to the empty sites that are present in the queued atoms,
        they therefore need to be reset

        Keyword Arguments:
            other_idxs (list | set | None): Other indexes that need to be reset
        """
        for idx_i in set(self.queued_atoms
                         + list(other_idxs if other_idxs is not None else [])):

            logger.info(f'Resetting sites on atom {idx_i}')

            atom = self.atoms[idx_i]
            atom.type.reset_onto(points=[self.atoms[idx].coord
                                         for idx in atom.neighbours],
                                 coord=atom.coord)
        return None

    def _adjust_ring_dihedrals(self, ring_bond, dihedrals):
        """Outsource the ring closure to an external function"""

        logger.info('Adjusting ring dihedrals to close the ring')

        start_time = time()
        coords = closed_ring_coords(py_coords=self.coordinates,
                                    py_curr_angles=dihedrals.values(self.atoms),
                                    py_ideal_angles=dihedrals.ideal_angles,
                                    py_axes=dihedrals.axes,
                                    py_rot_idxs=dihedrals.rot_idxs,
                                    py_origins=dihedrals.origins,
                                    py_rep_idxs=self.non_bonded_idx_matrix,
                                    py_close_idxs=np.array(tuple(ring_bond),
                                                           dtype='i4'),
                                    py_r0=ring_bond.r0)
        self.coordinates = coords
        logger.info(f'Closed ring in {(time() - start_time) * 1000:.2f} ms')
        return

    def _adjust_ring_angles(self, ring_bond):
        """Shift angles in a ring to close e.g. in a cyclopropane the 109º
        angles between carbons are much to large to generate a sensible
        geometry no matter the dihedral angles, so compress the C-C-C angles
        to 60º to close the ring e.g::


              C2---- C3                           C2
            /                     -->           /   |
           C1                                 C1 ---C3

        Arguments:


        """

        path = self._ring_path(ring_bond=ring_bond)

        angles_idxs = [tuple(path[i:i + 3]) for i in range(len(path) - 2)]
        logger.info(f'Adjusting {len(angles_idxs)} angles to close a ring')

        angles = Angles()

        for angle_idxs in angles_idxs:

            graph = self.graph.copy()
            graph.remove_edge(ring_bond[0], ring_bond[1])

            angle = Angle(idxs=angle_idxs,
                          phi0=(np.pi - (2.0 * np.pi / len(path))))

            try:
                angle.find_rot_idxs(graph=graph, atoms=self.atoms)

            except FailedToSetRotationIdxs:
                logger.warning(f'Could not adjust angle {angle_idxs}')

                # TODO: This could be more precise for fused rings
                continue

            angle_alt = Angle(idxs=angle_idxs,
                              rot_idxs=angle.inverse_rot_idxs(self.atoms),
                              phi0=angle.phi0)

            angles.append(angle)
            angles.append(angle_alt)

        coords = self.coordinates
        axes = []
        rot_idxs = [angle.rot_idxs for angle in angles]

        for i, angle in enumerate(angles):
            idx_x, idx_y, idx_z = angle.idxs

            axis = np.cross(coords[idx_x, :] - coords[idx_y, :],
                            coords[idx_z, :] - coords[idx_y, :])

            # Alternate between forward and reverse rotations
            if i % 2 == 0:
                axis *= -1

            # Append the axis onto the coordinates
            coords = np.concatenate((coords,
                                     np.expand_dims(axis + coords[idx_y, :], axis=0)))

            # Now the axis is coords[-1] - coods[idx_y], so
            axes.append([coords.shape[0] - 1, idx_y])

            # Append zeros to the rotated indexes corresponding to the
            # number of added coordinates (axes)
            rot_idxs[i] += len(angles) * [0]

        coords = rotate(py_coords=coords,
                        py_angles=angles.dvalues(self.atoms) / 2,
                        py_axes=np.array(axes, dtype='i4'),
                        py_rot_idxs=np.array(rot_idxs, dtype='i4'),
                        py_origins=angles.origins)

        self.coordinates = coords[:-len(angles), :]
        return

    def _close_ring(self, ring_bond):
        """
        Adjust ring dihedrals such that a ring is formed

        Arguments:
            ring_bond (autode.smiles.SMILESBond):
        """
        logger.info(f'Closing ring on: {ring_bond} and adjusting atoms')

        dihedrals = Dihedrals()
        for dihedral in self._ring_dihedrals(ring_bond):

            # Generate a graph without the ring or this dihedral to locate
            # the indexes that should be rotated
            graph = self.graph.copy()
            graph.remove_edge(*ring_bond)

            try:
                dihedral.find_rot_idxs(graph=graph, atoms=self.atoms)

            except FailedToSetRotationIdxs:
                logger.warning(f'Could not rotate dihedral {dihedral} '
                               f'splitting across {dihedral.mid_idxs} did not '
                               f'afford two fragments')
                continue

            dihedrals.append(dihedral)

        if len(dihedrals) == 0:
            logger.info('No dihedrals to adjust to close the ring')

        else:
            self._adjust_ring_dihedrals(ring_bond, dihedrals=dihedrals)

        # Calculate the bond length of the closing bond in the new coords
        r_c = np.linalg.norm(self.atoms[ring_bond[0]].coord
                             - self.atoms[ring_bond[1]].coord)

        if len(dihedrals) < 3 and not np.isclose(r_c, ring_bond.r0, atol=0.2):
            logger.info('Have a small ring that was poorly closed '
                        f'(r = {r_c:.2f} Å) - adjusting angles')
            self._adjust_ring_angles(ring_bond)

        self._reset_queued_atom_sites(other_idxs=ring_bond)
        return None

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
        logger.info('Minimising non-bonded repulsion by dihedral rotation')
        start_time = time()

        dihedrals = Dihedrals()

        for bond in self.bonds:

            if bond.order != 1:
                continue

            # Check that both atoms that form this bond have > 1 neighbours,
            # thus define a dihedral
            idx_x, idx_y = bond

            # Find the other atoms that form the 4 atom tuple
            try:
                idx_w = next(idx for idx in self.atoms[idx_x].neighbours
                             if idx != idx_y and self.atoms[idx].n_bonded > 1)
                idx_z = next(idx for idx in self.atoms[idx_y].neighbours
                             if idx != idx_x and self.atoms[idx].n_bonded > 1)

            except StopIteration:
                continue   # No suitable neighbours

            dihedral = Dihedral(idxs=[idx_w, idx_x, idx_y, idx_z])

            try:
                dihedral.find_rot_idxs(self.graph.copy(), atoms=self.atoms)

            except FailedToSetRotationIdxs:
                continue  # Bond could be in a ring etc.

            dihedrals.append(dihedral)

        if len(dihedrals) == 0:
            return  # No rotation required

        logger.info(f'Have {len(dihedrals)} dihedrals to rotate')
        logger.info(f'Populated list in {(time() - start_time)*1000:.2f} ms')

        coords = rotate(py_coords=self.coordinates,
                        py_angles=np.zeros(len(dihedrals)),
                        py_axes=dihedrals.axes,
                        py_rot_idxs=dihedrals.rot_idxs,
                        py_origins=dihedrals.origins,
                        minimise=True,
                        py_rep_idxs=self.non_bonded_idx_matrix)

        self.coordinates = coords
        logger.info(f'Performed final dihedral rotation in '
                    f'{(time() - start_time)*1000:.2f} ms')
        return None

    def _set_atoms_bonds(self, atoms, bonds):
        """
        From a list of SMILESAtoms, and SMILESBonds set the required attributes
        and convert all implicit hydrogens into explicit atoms
        """
        if atoms is None or len(atoms) == 0:
            raise SMILESBuildFailed('Cannot build a structure with no atoms')

        # Set attributes
        self.atoms, self.bonds = atoms, bonds
        self.graph = nx.Graph()
        self.queued_atoms = []
        self.queued_dihedrals = Dihedrals()

        self._explicit_all_hydrogens()

        # Add nodes for all the atom indexes, without attributes for e.g
        # atomic symbol as a normal molecular graph would have
        for i in range(self.n_atoms):
            self.graph.add_node(i)

        # Set the ideal bond lengths and the graph edges
        for bond in self.bonds:
            idx_i, idx_j = bond
            self.graph.add_edge(idx_i, idx_j, order=bond.order)

            bond.r0 = get_avg_bond_length(self.atoms[idx_i].label,
                                          self.atoms[idx_j].label)

        self._set_atom_types()

        # Add the first atom to the queue of atoms to be translated etc.
        self.queued_atoms.append(0)
        # perturb the first atom's coordinate slightly
        self.atoms[0].translate(vec=np.array([0.001, 0.001, 0.001]))
        return None

    def _queue_double_bond_dihedral(self, bond):
        """
        For a double bond queue the dihedral rotation to be applied such that::

              X -----Y
             /       |
            W        Z

        where the dihedral is 0 or π, depending on the stereochemistry

        Arguments:
            bond (autode.smiles.base.SMILESBond):
        """
        idx_x, idx_y = bond

        nbrs_x = [idx for idx in self.atoms[idx_x].neighbours if idx != idx_y]
        nbrs_y = [idx for idx in self.atoms[idx_y].neighbours if idx != idx_x]

        if len(nbrs_x) + len(nbrs_y) < 4:
            logger.info('Had a double bond with fewer than 4 neighbours - no '
                        'need to rotate the dihedral')
            return

        # Index W is the closest atom index to X, that isn't Y
        idx_w = nbrs_x[np.abs(np.array(nbrs_x) - idx_x).argmin()]
        # and similarly for Z
        idx_z = nbrs_y[np.abs(np.array(nbrs_y) - idx_y).argmin()]

        # Is this bond cis or trans?
        stro_x, stro_y = self.atoms[idx_x].stereochem, self.atoms[idx_y].stereochem

        phi = np.pi # Default to a trans double bond

        if ((all(self.atoms[idx].in_ring for idx in (idx_w, idx_x, idx_y, idx_z))
            and not self.atoms[idx_x].has_stereochem)
            or stro_x == stro_y == 'al_up'
            or stro_x == stro_y == 'al_down'):
            phi = 0

        dihedral = Dihedral([idx_w, idx_x, idx_y, idx_z], phi0=phi)

        try:
            dihedral.find_rot_idxs(graph=self.graph.copy(),
                                   atoms=self.atoms)
        except FailedToSetRotationIdxs:
            if self.atoms[idx_x].stereochem is not None:
                logger.error(f'Could not queue {dihedral} for {bond} E/Z'
                             f'stereochemistry may be wrong')

            else:
                logger.warning(f'Could not queue {dihedral} for {bond}')

            return

        logger.info(f'Queuing {dihedral}')
        self.queued_dihedrals.append(dihedral)
        return None

    def _rotate_dihedrals(self):
        """Rotate all dihedrals in the queue"""
        if len(self.queued_dihedrals) == 0:
            return   # Nothing to be done

        logger.info(f'Have {len(self.queued_dihedrals)} dihedral(s) to rotate')

        dphis = [dihedral.phi0 - dihedral.value(self.atoms)
                 for dihedral in self.queued_dihedrals]

        self.coordinates = rotate(py_coords=self.coordinates,
                                  py_angles=np.array(dphis, dtype='f8'),
                                  py_axes=self.queued_dihedrals.axes,
                                  py_rot_idxs=self.queued_dihedrals.rot_idxs,
                                  py_origins=self.queued_dihedrals.origins)

        self.queued_dihedrals.clear()
        self._reset_queued_atom_sites()
        return None

    def _add_bonded_atoms(self, idx):
        """
        Add all the atoms bonded to a particular index, that have not already
        been shifted

        Arguments:
            idx (int): Atom index
        """
        atom = self.atoms[idx]

        for bond in self.bonds.involving(idx):
            bonded_idx = bond[0] if bond[1] == idx else bond[1]

            if bonded_idx in self.queued_atoms:
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
                site = atom.type.empty_site_mr(atom.coord,
                                               other_coords=self.coordinates)

            # Coordinate of this atom is the current position shifted by
            # the ideal distance in a direction of a empty coordination
            # site on the atom
            coord = bond.r0 * site + atom.coord
            bonded_atom = self.atoms[bonded_idx]
            bonded_atom.translate(coord)

            # Atoms that are not terminal need to be added to the queue
            if not isinstance(self.atoms[bonded_idx].type, TerminalAtom):
                # and the atom type rotated so an empty site is coincident
                # with this atom
                bonded_atom.type.rotate_empty_onto(point=atom.coord,
                                                   coord=bonded_atom.coord)
                # and queue
                self.queued_atoms.append(bonded_idx)

        return None

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
        start_time = time()
        self._set_atoms_bonds(atoms, bonds)

        while not self.built:

            idx = self.queued_atoms.pop(0)
            self._add_bonded_atoms(idx)
            self._rotate_dihedrals()

            logger.info(f'Queue: {self.queued_atoms}')

        self._minimise_non_ring_dihedrals()
        logger.info(f'Built 3D in {(time() - start_time)*1E3:.2f} ms')
        return None

    def __init__(self):
        """
        Coordinate builder initialised from a set of atoms and bonds connecting
        them. This builder should generate something *reasonable* that can
        be cleaned up with a forcefield
        """
        self.atoms = None              # list(SMILESAtom)
        self.bonds = None              # SMILESBonds
        self.graph = None              # nx.Graph
        self.rings_idxs = None         # Iterator for atom indexes in all rings

        # A queue of atom indexes, the neighbours for which need to be added
        self.queued_atoms = []

        # A queue of dihedrals that need to be applied
        self.queued_dihedrals = Dihedrals()


class AtomType:

    def empty_site(self):
        """Iterator for the coordinate of the next free site"""
        return self._site_coords.pop(0)

    def empty_site_mr(self, point, other_coords):
        """Return the site on this atom that is furthest from all other
        coordinates using a simple 1/r potential where r is the distance from
        the site to the other coordinates

        Arguments:
            point (np.ndarray): Coordinate of this atom, shape = (3,)
            other_coords (np.ndarray): Other coordinates, shape = (N, 3)

        Returns:
            (np.ndarray): Coordinate of the site centered at the origin
        """
        dists = np.array([np.linalg.norm(other_coords - (site + point), axis=1)
                          for site in self._site_coords])

        repulsion = np.sum(np.power(dists, -1), axis=1)
        return self._site_coords.pop(np.argmin(repulsion))

    def reset_onto(self, points, coord):
        """
        Reset the site coordinates given a set of points. Ignore any points
        located exactly at the origin and, once fitted, remove the sites
        that are coincident with the points

        Arguments:
            points (iterable(np.ndarray)): List (or iterable) of points that
                   that the sites need to be reset onto
            coord (np.ndarray): Coordinate of this atom
        """
        origin = np.zeros(3)
        points = np.array([(point - coord) / np.linalg.norm(point - coord)
                           for point in points
                           if not np.allclose(point, origin)])

        # Take a copy of the template coordinates to rotate and delete
        site_coords = np.copy(self.template_site_coords)

        logger.info(f'Rotating {len(site_coords)} sites onto'
                    f' {len(points)} points')

        # Rotate all the sites such that n sites are optimally orientated onto
        # the (fixed) points
        rot_mat = get_rot_mat_kabsch(p_matrix=site_coords[:len(points)],
                                     q_matrix=points)

        site_coords = np.dot(rot_mat, site_coords.T).T

        # For each point (row) calculate the minimum distance to a site on
        # this atom
        min_dists = np.min(distance_matrix(site_coords, points), axis=1)

        # Re-populate the empty sites, which are the sites that are not the
        # closest to the points
        self._site_coords = [coord for i, coord in enumerate(site_coords)
                             if i not in np.argsort(min_dists)[:len(points)]]
        return None

    def rotate_empty_onto(self, point, coord):
        """Rotate the site coordinates such that an empty site is coincident
        with the vector from a coordinate to a point, and remove the site
        from the list of available sites"""
        return self.rotate_onto(point, coord, site=self.empty_site())

    def rotate_randomly(self):
        """Rotate the sites randomly to prevent zero cross products"""
        point = np.copy(self._site_coords[0])
        point += np.random.uniform(0.01, 0.02, size=3)

        self.rotate_onto(point=point, coord=np.zeros(3),
                         site=self._site_coords[0])
        return

    def rotate_onto(self, point, coord, site):
        """
        Rotate this atom type so a site is coincident with a point if this
        atom is at a coord i.e.

                           site
                           /
                          /         -->
        point--------coord                 point--site--coord


        -----------------------------------------------------------------------
        Arguments:
            point (np.ndarray): shape = (3,)

            coord (np.ndarray): shape = (3,)

            site (np.ndarray): shapte = (3,)
        """
        vector = point - coord

        normal = np.cross(site, vector)
        normal /= np.linalg.norm(normal)

        # Sites are normal vectors, no no need for mod
        arg = np.dot(site, vector) / np.linalg.norm(vector)

        # cos(-θ/2) = √(arg + 1) / √2
        # sin(-θ/2) = √(1-arg) / √2
        a = np.sqrt(1.0 + arg) / np.sqrt(2)
        b, c, d = -normal * (np.sqrt(1.0 - arg) / np.sqrt(2))

        # 3D rotation matrix from the Euler–Rodrigues formula
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rot_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                               [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                               [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

        # Rotate all the sites (no need to translate as they're already
        # positioned around the origin)
        self._site_coords = [np.matmul(rot_matrix, site)
                             for site in self._site_coords]
        return None

    def __init__(self, site_coords, is_chiral=False):
        """Base atom type class

        Arguments:
            site_coords (list(np.ndarray)): Shape = (n, 3) should contain a
                        list of unit vectors pointing in directions where other
                        atoms can be added

        Keyword Arguments:
            is_chiral (bool): Is this atom type chiral e.g. a tetrahedral atom
                              with four different substituents
        """
        self.template_site_coords = np.copy(site_coords)
        self._site_coords = site_coords

        self.is_chiral = is_chiral
        self.rotate_randomly()


class TerminalAtom(AtomType):

    def __init__(self):
        """
        Terminal atom with a site pointing along the x-axis::

                    Atom--->
        """
        site_coords = [np.array([1.0, 0.0, 0.0])]

        super().__init__(site_coords)


class LinearAtom(AtomType):

    def __init__(self):
        """
        Linear atom with sites pointing along the x-axis::

                    <---Atom--->
        """
        site_coords = [np.array([1.0, 0.0, 0.0]),
                       np.array([-1.0, 0.0, 0.0])]

        super().__init__(site_coords)


class BentAtom(AtomType):

    def __init__(self):
        """
        Bent atom with sites generated by optimisation of H2O::

                       Atom
                     /     \
        """
        site_coords = [np.array([-0.78226654, -0.62294387, 0.0]),
                       np.array([0.78322832, -0.62173419, 0.0])]

        super().__init__(site_coords)


class TrigonalPyramidalAtom(AtomType):

    def __init__(self):
        """
        Trigonal pyramidal atom e.g. P in PH3, obtained from optimisation of
        ammonia (then normalising  NH distances to 1 Å)::

                      Atom
                    /  |  \
        """
        site_coords = [np.array([0.90023489, -0.14794295, -0.40949973]),
                       np.array([-0.58738609, -0.70512041, -0.39721881]),
                       np.array([-0.32432922, 0.85865859, -0.39688283])]

        super().__init__(site_coords)


class TrigonalAtom(AtomType):

    def __init__(self):
        """
        Trigonal atom e.g. [CR3]+ , obtained from optimisation of BH3
        (then normalising  NH distances to 1 Å)::

                               /
                       --- Atom
                              \
        """
        site_coords = [np.array([-0.506363095, -0.862320319, 0.0]),
                       np.array([-0.495155944, 0.868804058, 0.0]),
                       np.array([0.999977780, -0.006666131, 0.0])]

        super().__init__(site_coords)


class TetrahedralAtom(AtomType):

    def __init__(self):
        """
        Tetrahedral atom with sites generated by optimisation of methane
        (then normalising CH distances to 1 Å)::

                         |  /
                       Atom
                     /     \

        """
        site_coords = [np.array([-0.404709,  0.86798519, -0.28777090]),
                       np.array([-0.580775, -0.75435372, -0.30602419]),
                       np.array([0.0763827, -0.01927872,  0.99689218]),
                       np.array([0.9089159, -0.09390161, -0.40626889])]

        super().__init__(site_coords)


class TetrahedralNAtom(TetrahedralAtom):
    """A 'normal' order chiral tetrahedral atom"""

    def __init__(self):
        super().__init__()
        self.is_chiral = True


class TetrahedralIAtom(TetrahedralAtom):
    """An 'inverted' order chiral tetrahedral atom"""

    def empty_site(self):
        """Swap the first two yielded site coordinates, effectively swapping
        the chirality this atom's neighbours are added"""

        if len(self._site_coords) == 3:
            return self._site_coords.pop(1)

        else:
            return super().empty_site()

    def __init__(self):
        super().__init__()
        self.is_chiral = True


class SquarePlanarAtom(AtomType):

    def __init__(self):
        """
        Square planar atom with sites generated by optimisation of XeF4::

                       |
                   -- Atom --
                       |
        """
        site_coords = [np.array([-0.99779169,  0.06642094,  0.0]),
                       np.array([0.06641523, 0.99779207, 0.0]),
                       np.array([ 0.99779219, -0.06641349,  0.0]),
                       np.array([-0.06642889, -0.99779116,  0.0])]

        super().__init__(site_coords)


class TrigonalBipyramidalAtom(AtomType):

    def __init__(self):
        """
        Trigonal bipyramidal atom with sites generated by optimisation of
        [Cn(Cl)5]-::

                        |
                        |
                  --- Atom --
                    /  |
                       |
        """
        site_coords = [np.array([-0.96060076, 0.0333159, -0.27592795]),
                       np.array([0.27025683, -0.50748281, 0.8181824]),
                       np.array([0.68796654, 0.4787655, -0.54542243]),
                       np.array([-0.13392744,  0.82131359, 0.55453352]),
                       np.array([0.13543747, -0.82200953, -0.55313383])]

        super().__init__(site_coords)


class OctahedralAtom(AtomType):

    def __init__(self):
        """
        Octahedral atom with sites generated by optimisation of [Co(Cl)6]3-::

                        |
                        | /
                  --- Atom --
                    /  |
                       |
        """
        site_coords = [np.array([0.06037748, 0.86107926, 0.50487332]),
                       np.array([-0.96772781, -0.0717284, 0.24157384]),
                       np.array([-0.06059905, -0.86084488, -0.50524632]),
                       np.array([-0.2330902, 0.50187662, -0.83293986]),
                       np.array([0.23246809, -0.50140995, 0.83339465]),
                       np.array([0.96764951, 0.07148532, -0.24195925])]

        super().__init__(site_coords)


class PentagonalBipyramidalAtom(AtomType):

    def __init__(self):
        """Approximate trigonal pentagonal geometry by optimisation of IF7"""

        site_coords = [np.array([-0.82513358, 0.19948399, 0.52854584]),
                       np.array([0.36100434, 0.82474278, 0.4352875]),
                       np.array([-0.2989535, - 0.86729114, -0.39803628]),
                       np.array([0.07965322, - 0.50323863, 0.86046862]),
                       np.array([-0.68679889, 0.35535251, -0.63405984]),
                       np.array([0.92361199, - 0.38311825, 0.0127003]),
                       np.array([0.4702203,  0.36157889, -0.80507986])]

        super().__init__(site_coords)


class SquareAntiprismAtom(AtomType):

    def __init__(self):
        """
        Approximate square antiprism geometry by optimisation of [XeF8]2-
        """
        site_coords = [np.array([-0.12556124, 0.56801979, -0.81338053]),
                       np.array([-0.9236697, -0.18379172, -0.33623635]),
                       np.array([-0.13553555, -0.94631881, 0.29344647]),
                       np.array([0.19928886, -0.57504162, -0.79348036]),
                       np.array([0.54323185, -0.11585026, 0.83155148]),
                       np.array([-0.62763462, 0.12547974, 0.76832911]),
                       np.array([0.10138916, 0.94996354, 0.29544797]),
                       np.array([0.95080191, 0.18740602, -0.24668749])]

        super().__init__(site_coords)


class Angle:

    def __str__(self):
        return f'Angle(idxs={self.idxs})'

    def __repr__(self):
        return self.__str__()

    def value(self, atoms):
        """

        Args:
            atoms (list(autode.atoms.Atom)):

        Returns:
            (float): Angle in radians
        """

        idx_x, idx_y, idx_z = self.idxs
        vec1 = atoms[idx_x].coord - atoms[idx_y].coord
        vec2 = atoms[idx_z].coord - atoms[idx_y].coord

        return np.arccos(np.dot(vec1, vec2)
                         / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def _find_rot_idxs_from_pair(self, graph, atoms, pair):
        """
        Split the graph across a pair of indexes and set the atom indexes
        to be rotated

        Arguments:
            graph (nx.Graph):
            atoms (list(autode.atoms.Atom)):
            pair (list(int)): len == 2
        """
        graph.remove_edge(*pair)

        # Remove all the nodes in the graph that have not been shifted, thus
        # the rotation indexes only include atoms that have been 'built'
        for idx, atom in enumerate(atoms):
            if hasattr(atom, 'is_shifted') and not atom.is_shifted:
                graph.remove_node(idx)

        components = [graph.subgraph(c) for c in
                      nx.connected_components(graph)]

        if len(components) != 2:
            raise FailedToSetRotationIdxs(f'Splitting over {pair} did '
                                          'not afford two fragments')

        # Choose the components that will be rotated
        cpnt_idx = 0 if pair[0] in components[0].nodes else 1

        self.rot_idxs = [1 if i in components[cpnt_idx].nodes else 0
                         for i in range(len(atoms))]
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

        Returns:
            (list(int)):
        """
        return [1 if (hasattr(atom, 'is_shifted') and atom.is_shifted)
                and self.rot_idxs[i] != 1 else 0
                for i, atom in enumerate(atoms)]

    @property
    def phi0(self):
        """A non-None ideal angle, default to 100 degrees"""
        return 1.74533 if self.phi_ideal is None else self.phi_ideal

    def __init__(self, idxs, rot_idxs=None, phi0=None):
        """Angle between a set of atoms. In order"""

        self.idxs = idxs
        self.phi_ideal = phi0
        self.rot_idxs = rot_idxs


class Angles(list):

    @property
    def axes(self):
        raise NotImplementedError

    @property
    def origins(self):
        """Origins for the rotation, as the central atom of the trio"""
        return np.array([angle.idxs[1] for angle in self], dtype='i4')

    @property
    def rot_idxs(self):
        """Matrix of atom indexes to rotate"""
        return np.array([angle.rot_idxs for angle in self], dtype='i4')

    @property
    def ideal_angles(self):
        """Ideal angle vector (float | None)"""
        return [angle.phi_ideal for angle in self]

    def values(self, atoms):
        """Current angle vector in radians"""
        return np.array([angle.value(atoms) for angle in self], dtype='f8')

    def dvalues(self, atoms):
        """Difference between the current and ideal angles"""
        return np.array([angle.phi0 - angle.value(atoms) for angle in self],
                        dtype='f8')


class Dihedrals(Angles):

    @property
    def axes(self):
        return np.array([dihedral.mid_idxs for dihedral in self], dtype='i4')

    @property
    def origins(self):

        origins = []
        for dihedral in self:
            idx_i, idx_j = dihedral.mid_idxs
            origins.append(idx_i if dihedral.rot_idxs[idx_i] == 1 else idx_j)

        return np.array(origins, dtype='i4')


class Dihedral(Angle):
    """A dihedral defined by 4 atom indexes e.g.

       X       W
       |      /
       Y---- Z
    """
    def __str__(self):
        return f'Dihedral(idxs={self.idxs})'

    @property
    def phi0(self):
        """A non-None ideal angle for this dihedral"""
        return 0.0 if self.phi_ideal is None else self.phi_ideal

    def value(self, atoms):
        """
        Calculate the value of a dihedral defined by some atoms with non-zero
        positions

        Arguments:
            atoms (list(autode.atoms.Atom)):

        Returns:
            (float): The dihedral angle in radians
        """
        idx_x, idx_y, idx_z, idx_w = self.idxs

        vec_yx = atoms[idx_x].coord - atoms[idx_y].coord
        vec_zw = atoms[idx_w].coord - atoms[idx_z].coord
        vec_yz = atoms[idx_z].coord - atoms[idx_y].coord

        vec1 = np.cross(vec_yx, vec_yz)
        vec2 = np.cross(-vec_yz, vec_zw)

        zero_vec = np.zeros(3)
        if np.allclose(vec1, zero_vec) or np.allclose(vec2, zero_vec):
            raise ValueError('Cannot calculate a dihedral - one zero vector')

        # Normalise everything
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)
        vec_yz /= np.linalg.norm(vec_yz)

        """
        Dihedral angles are defined as from the IUPAC gold book: "the torsion 
        angle between groups A and D is then considered to be positive if 
        the bond A-B is rotated in a clockwise direction through less than
        180 degrees"
        """
        angle = -np.arctan2(np.dot(np.cross(vec1, vec_yz), vec2),
                            np.dot(vec1, vec2))
        return angle

    def find_rot_idxs(self, graph, atoms):
        """
        Find the atom indexes that should be rotated for this dihedral

        Arguments:
            graph (nx.Graph):
            atoms (list(autode.atoms.Atom)):
        """
        return self._find_rot_idxs_from_pair(graph, atoms, pair=self.mid_idxs)

    def __init__(self, idxs, rot_idxs=None, phi0=None):
        """
        A dihedral constructed from atom indexes and possibly indexes that
        should be rotated, if this dihedral is altered::

             W
              \
               X --- Y
                     |
                     Z

        -----------------------------------------------------------------------
        Arguments:
            idxs (list(int)): 4 atom indexes defining the dihedral

        Keyword Arguments:
            rot_idxs (list(int) | None): Indexes to rotate, 1 if the atoms
                                         should be rotated else 0

            phi0 (float | None): Ideal angle for this dihedral (radians)
        """
        super().__init__(idxs=idxs, rot_idxs=rot_idxs, phi0=phi0)

        # Atom indexes of the central two atoms (X, Y)
        _, idx_y, idx_z, _ = idxs
        self.mid_idxs = (idx_y, idx_z)
