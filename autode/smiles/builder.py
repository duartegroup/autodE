import numpy as np
import networkx as nx
from time import time
from autode.log import logger
from autode.atoms import chalcogens, pnictogens
from autode.bonds import get_avg_bond_length
from autode.exceptions import SMILESBuildFailed
from autode.smiles.base import SMILESAtom, SMILESBond
from autode.smiles.rings import minimise_ring_energy


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

            # To build linear molecules the sites on atom types (templates)
            # need to be reversed for every other atom, otherwise the next atom
            # will be added to the same site forming a ring
            swap_order = i % 2 == 1

            if atom.n_bonded == 1:                              # e.g. H2, FCH3
                atom.type = TerminalAtom()

            elif atom.n_bonded == 2:                               # e.g. OH2, SR2
                if atom.label in chalcogens:
                    atom.type = BentAtom()

                else:                                         # e.g. AuR2
                    atom.type = LinearAtom()

            elif atom.n_bonded == 3:                               # e.g. NH3
                if atom.label in pnictogens:
                    atom.type = TrigonalPyramidalAtom()

                else:                                         # e.g. BH3
                    atom.type = TrigonalAtom()

            elif atom.n_bonded == 4:                               # e.g. CH4
                atom.type = TetrahedralAtom(swap_order=swap_order)

            else:
                raise NotImplementedError

        return None

    def _ring_dihedrals(self, ring_bond):
        """
        Given a ring bond find all the rotatable dihedrals that can be adjusted
        to close it with a reasonable bond distance

        Arguments:
            ring_bond (autode.smiles.SMILESBond):

        Returns:
            (iterator(tuple(int))):
        """
        # Indexes of atoms in the ring that should be closed
        try:
            ring_idxs = next(idxs for idxs in self.rings_idxs
                             if ring_bond[0] in idxs and ring_bond[1] in idxs)
        except StopIteration:
            raise SMILESBuildFailed(f'No ring containing {ring_bond}')

        # Find the path along which dihedrals can be defined e.g.
        #
        #         C2----C3
        #       /       |        -->  1, 2, 3, 4
        #     C1        C4
        #
        paths = nx.shortest_simple_paths(self.graph,
                                         source=ring_bond[0],
                                         target=ring_bond[1])
        path = None
        for possible_path in paths:

            # Can always have a path that traverses the ring bond (C1-C4 above)
            if len(possible_path) == 2:
                continue

            # For multiple fused rings there may be other paths that could be
            # traversed, so only take the one that has the appropriate idxs
            if all(idx in ring_idxs for idx in possible_path):
                path = possible_path
                break

        if path is None:
            raise SMILESBuildFailed('Could not find path in ring')

        # The dihedrals are then the all the 4 atom tuples in sequence
        dihedrals = [tuple(path[i:i + 4]) for i in range(len(path)- 3)]

        # so only add the indexes where the bond (edge) order is one
        for i, dihedral_idxs in enumerate(dihedrals):
            _, idx_i, idx_j, _ = dihedral_idxs

            if self.graph.get_edge_data(idx_i, idx_j)['order'] == 1:
                yield dihedral_idxs

    def _close_ring(self, ring_bond):
        """
        Adjust ring dihedrals such that a ring is formed

        Arguments:
            idxs (set(int)): Indexes of atoms that have been shifted from the
                             origin, and thus need to be shifted to rotate

            ring_bond (autode.smiles.SMILESBond):
        """
        logger.info(f'Closing ring with bond: {ring_bond} and adjusting atoms')

        pairs_rot_idxs = {}
        for dihedral in self._ring_dihedrals(ring_bond):
            _, idx_i, idx_j, _ = dihedral

            # Generate a graph without the ring or this dihedral to locate
            # the indexes that should be rotated
            graph = self.graph.copy()
            graph.remove_edge(idx_i, idx_j)
            graph.remove_edge(*ring_bond)

            components = [graph.subgraph(c)
                          for c in nx.connected_components(graph)]

            if len(components) != 2:
                logger.warning(f'Could not rotate dihedral {dihedral} '
                               f'splitting across {idx_i}-{idx_j} did not '
                               f'afford two fragments')
                continue

            # choose the first set of indexes (0) to rotate, this is arbitrary
            # but only rotate the atoms that have been shifted from the origin
            # (i.e. have been 'built')
            rot_idxs = [idx for idx in components[0].nodes
                        if not np.allclose(self.atoms[idx].coord, np.zeros(3))]

            pairs_rot_idxs[(idx_i, idx_j)] = rot_idxs

        minimise_ring_energy(atoms=self.atoms,
                             pairs_rot_idxs=pairs_rot_idxs,
                             close_idxs=(ring_bond[0], ring_bond[1]),
                             r0=ring_bond.r0)

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

        shifted_idxs = set()

        while not self.built:

            idx = self.queued_atoms[0]
            atom = self.atoms[idx]

            # Add all the atoms that are bonded to this one
            for bond in self.bonds.involving(idx):
                bonded_idx = bond[0] if bond[1] == idx else bond[1]

                if bonded_idx in shifted_idxs:
                    continue

                if bonded_idx in self.queued_atoms:
                    self._close_ring(ring_bond=bond)
                    continue

                # Coordinate of this atom is the current position shifted by
                # the ideal distance in a direction of a empty coordination
                # site on the atom
                coord = bond.r0 * atom.type.empty_site() + atom.coord
                bonded_atom = self.atoms[bonded_idx]
                bonded_atom.translate(coord)

                # Atoms that are not terminal need to be added to the queue
                if not isinstance(atoms[bonded_idx].type, TerminalAtom):

                    # and the atom type rotated so an empty site is coincident
                    # with this atom
                    bonded_atom.type.rotate_empty_onto(point=atom.coord,
                                                       coord=bonded_atom.coord)

                    # and queue
                    self.queued_atoms.append(bonded_idx)

            # And remove this atom from the queue and added to shifted
            self.queued_atoms.remove(idx)
            shifted_idxs.add(idx)
            logger.info(f'Queue: {self.queued_atoms}')

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


class AtomType:

    def empty_site(self):
        """Iterator for the coordinate of the next free site"""
        return self._site_coords.pop(0)

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

    def __init__(self, site_coords):
        """Base atom type class

        Arguments:
            site_coords (list(np.ndarray)): Shape = (n, 3) should contain a
                        list of unit vectors pointing in directions where other
                        atoms can be added
        """
        self._site_coords = site_coords
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

    def __init__(self, swap_order=False):
        """
        Tetrahedral atom with sites generated by optimisation of methane
        (then normalising CH distances to 1 Å)::

                         |  /
                       Atom
                     /     \


        Keyword Arguments:
            swap_order (bool): Reverse the order of the sites?
        """
        site_coords = [np.array([-0.580775, -0.75435372, -0.30602419]),
                       np.array([-0.404709,  0.86798519, -0.28777090]),
                       np.array([0.0763827, -0.01927872,  0.99689218]),
                       np.array([0.9089159, -0.09390161, -0.40626889])]

        if swap_order:
            site_coords = site_coords[::-1]

        super().__init__(site_coords)
