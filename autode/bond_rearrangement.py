from itertools import combinations, combinations_with_replacement
from typing import Optional, Iterator, TYPE_CHECKING
import os

from autode.geom import get_neighbour_list
from autode.log import logger
from autode.config import Config
from autode.mol_graphs import (
    get_bond_type_list,
    get_fbonds,
    is_isomorphic,
    find_cycles,
)

if TYPE_CHECKING:
    from autode.species.species import Species
    import networkx as nx


_MAX_BOND_REARR = 4  # maximum allowed bond rearrangment


class MoveSet:
    """A set of moves with bond types, breaking and forming"""

    def __init__(
        self,
        fbond_types: list[tuple[str, int]],
        bbond_types: list[tuple[str, int]],
    ):
        """

        Args:
            fbond_types (list[tuple]): A list of bond types forming
                    with numbers of each e.g. [('CH', 2), ('OH', 4)]
            bbond_types (list[tuple]): A list of bond types breaking
                    with numbers of each
        """
        assert all(
            isinstance(fb_type, tuple)
            and isinstance(fb_type[0], str)
            and isinstance(fb_type[1], int)
            for fb_type in fbond_types
        )
        assert all(
            isinstance(bb_type, tuple)
            and isinstance(bb_type[0], str)
            and isinstance(bb_type[1], int)
            for bb_type in bbond_types
        )
        self.fbonds = fbond_types
        self.bbonds = bbond_types


class BondRearrGenerator:
    def __init__(
        self,
        reactant: "Species",
        product: "Species",
        delta_bond_tot: int,
        extra_move_pairs: int = 0,
    ):
        """
        Create a class that obtains all possible bond rearrangements that
        will transform the reactant into the product based on the types of
        bonds that must be broken and formed

        Args:
            reactant:
            product:
            delta_bond_tot: Total number of bonds changing from reactant
                            to product
            extra_move_pairs: Number of pairs of extra graph moves that can
                            be made in addition to the known types
        """
        self._reactant_bond_dict: Optional[dict[str, list]] = None

        self._reactant = reactant
        self._product = product
        assert isinstance(delta_bond_tot, int)

        # number of extra *pairs* of bond moves (form + break)
        assert extra_move_pairs >= 0
        self._n_extra_pair = int(extra_move_pairs)

        # every moveset contains types and numbers of breaking bonds
        # as well as forming bonds
        self._movesets: list[MoveSet] = []
        self._determine_bond_types(delta_bond_tot)

    def _determine_bond_types(self, delta_bond_tot: int) -> None:
        """
        Determine the types and numbers of bonds that must be broken and
        formed based on the reactant and product

        Args:
            delta_bond_tot: Total number of bonds changing from reactant
                            to product
        """
        self._reactant_bond_dict = get_bond_type_list(self._reactant.graph)
        product_bond_dict = get_bond_type_list(self._product.graph)

        total_delta = 0
        known_bbond_types = {}
        known_fbond_types = {}
        # First handle bonds of known types that must form/break
        for reac_key, reac_bonds in self._reactant_bond_dict.items():
            prod_bonds = product_bond_dict[reac_key]
            delta_bonds = len(prod_bonds) - len(reac_bonds)
            total_delta += delta_bonds
            if delta_bonds > 0:
                known_fbond_types[reac_key] = delta_bonds
            elif delta_bonds < 0:
                known_bbond_types[reac_key] = -delta_bonds
        assert total_delta == delta_bond_tot, "Bond types do not match!"

        if self._n_extra_pair == 0:
            if len(known_bbond_types) + len(known_fbond_types) != 0:
                self._movesets.append(
                    MoveSet(
                        fbond_types=list(known_fbond_types.items()),
                        bbond_types=list(known_bbond_types.items()),
                    )
                )
            return None

        # Extra pairs of bonds to break and form must be from bonds of type
        # which are present in both reactant and product
        common_types = []
        for b_key in self._reactant_bond_dict.keys():
            if (
                len(self._reactant_bond_dict[b_key]) > 0
                and len(product_bond_dict[b_key]) > 0
            ):
                common_types.append(b_key)

        type_combs = list(
            combinations_with_replacement(common_types, self._n_extra_pair)
        )

        for comb in type_combs:
            this_bbond_types = known_bbond_types.copy()
            this_fbond_types = known_fbond_types.copy()
            for key in comb:
                this_bbond_types[key] = this_bbond_types.get(key, 0) + 1
                this_fbond_types[key] = this_fbond_types.get(key, 0) + 1
            self._movesets.append(
                MoveSet(
                    fbond_types=list(this_fbond_types.items()),
                    bbond_types=list(this_bbond_types.items()),
                )
            )

        self._sanitise_movesets()
        return None

    def _sanitise_movesets(self) -> None:
        """
        Check that all the movesets are actually appropriate and
        are not forming/breaking more bonds than is actually possible
        """
        assert self._reactant_bond_dict is not None
        new_movesets = []
        for moveset in self._movesets:
            is_valid = True
            for bbond_type, num in moveset.bbonds:
                if len(self._reactant_bond_dict[bbond_type]) < num:
                    is_valid = False
                    break
            for fbond_type, num in moveset.fbonds:
                if len(get_fbonds(self._reactant.graph, fbond_type)) < num:
                    is_valid = False
                    break
            if is_valid:
                new_movesets.append(moveset)

        self._movesets = new_movesets
        return None

    def _graph_edits(
        self,
        moveset: MoveSet,
        bbonds: Optional[tuple] = None,
        fbonds: Optional[tuple] = None,
        counter: int = 0,
    ) -> Iterator[list[tuple]]:
        """
        Generator that yields all possible forming and breaking bonds
        based on the allowed graph moves defined in moveset. Only the
        moveset should be supplied, other arguments are used to keep
        track of the internal state of the generator.

        Args:
            moveset: A tuple of two lists, each of the type [("CH", 2), ...],
                     first being the types and numbers of bonds to be broken
                     and the second being those to be formed
            bbonds: (State variable) tuple of bonds broken so far
            fbonds: (State variable) tuple of bonds formed so far
            counter: (State variable) number of bond types operated

        Yields:
            (list[tuple]): A list with two items, the first being
                        the bonds being broken, and the second being
                        the bonds being formed
        """
        assert self._reactant.graph is not None
        assert self._reactant_bond_dict is not None
        assert len(moveset.bbonds) + len(moveset.fbonds) != 0

        if counter == 0:
            bbonds, fbonds = tuple(), tuple()
        else:
            assert bbonds is not None and fbonds is not None

        # check if all graph edits have been made already
        if counter == (len(moveset.fbonds) + len(moveset.bbonds)):
            yield [bbonds, fbonds]
            return

        # First, break bonds
        if counter < len(moveset.bbonds):
            bbond_type, num = moveset.bbonds[counter]
            possible_bbonds = self._reactant_bond_dict[bbond_type]
            counter += 1
            for comb in combinations(possible_bbonds, num):
                yield from self._graph_edits(
                    moveset, bbonds + tuple(comb), fbonds, counter
                )

        # Next, form bonds
        else:
            counter_f = counter - len(moveset.bbonds)
            fbond_type, num = moveset.fbonds[counter_f]
            possible_fbonds = get_fbonds(self._reactant.graph, fbond_type)
            counter += 1
            for comb in combinations(possible_fbonds, num):
                yield from self._graph_edits(
                    moveset, bbonds, fbonds + tuple(comb), counter
                )

    def get_valid_bond_rearrs(self) -> list["BondRearrangement"]:
        """
        Get all bond rearrangements that are actually valid, i.e. transforms
        the reactant to product based on graph isomorphism

        Returns:
            (list(BondRearrangement)):
        """
        valid_bond_rearrs: list["BondRearrangement"] = []
        for moveset in self._movesets:
            for bbonds, fbonds in self._graph_edits(moveset):
                valid_bond_rearrs = add_bond_rearrangment(
                    valid_bond_rearrs,
                    self._reactant,
                    self._product,
                    fbonds,
                    bbonds,
                )
        return valid_bond_rearrs


def get_bond_rearrangs(reactant, product, name, save=True):
    """For a reactant and product (mol_complex) find the set of breaking and
    forming bonds that will turn reactants into products. This works by
    determining the types of bonds that have been made/broken (i.e CH) and
    then only considering rearrangements involving those bonds.

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.species.ReactantComplex):

        product (autode.species.ProductComplex):

        name (str):

    Keyword Arguments:
        save (bool): Save bond rearrangements to a file for fast reloading

    Returns:
        (list(autode.bond_rearrangements.BondRearrangement)):
    """
    logger.info(f"Finding the possible forming and breaking bonds for {name}")

    if os.path.exists(f"{name}_bond_rearrangs.txt"):
        return get_bond_rearrangs_from_file(f"{name}_bond_rearrangs.txt")

    if is_isomorphic(reactant.graph, product.graph) and product.n_atoms > 3:
        logger.error(
            "Reactant (complex) is isomorphic to product (complex). "
            "Bond rearrangement cannot be determined unless the "
            "substrates are limited in size"
        )
        return None

    # The change in the number of bonds should be < 0 as in the reaction
    # initialisation reacs/prods are swapped otherwise
    delta_n_bonds = (
        product.graph.number_of_edges() - reactant.graph.number_of_edges()
    )
    abs_delta = abs(delta_n_bonds)

    if delta_n_bonds > 0:
        logger.warning(
            "More bonds in product than in reactant,"
            "suggest swapping them for better TS search"
        )
    if abs_delta > _MAX_BOND_REARR:
        logger.error(
            "Cannot treat a change in bonds reactant <-> product of"
            f" {delta_n_bonds}"
        )
        return None

    max_extra_pairs = (_MAX_BOND_REARR - abs_delta) // 2

    for extra_pairs in range(max_extra_pairs + 1):
        bond_gen = BondRearrGenerator(
            reactant, product, delta_n_bonds, extra_pairs
        )
        possible_brs = bond_gen.get_valid_bond_rearrs()

        if len(possible_brs) > 0:
            logger.debug(f"Found molecular graph rearrangement(s) to products")

            # This function will return with the first bond rearrangement
            # that leads to products
            n_bond_rearrangs = len(possible_brs)
            if n_bond_rearrangs > 1:
                logger.info(
                    f"Multiple *{n_bond_rearrangs}* possible bond "
                    f"breaking/makings are possible"
                )
                possible_brs = strip_equiv_bond_rearrs(possible_brs, reactant)
                prune_small_ring_rearrs(possible_brs, reactant)

            if save:
                save_bond_rearrangs_to_file(
                    possible_brs, filename=f"{name}_BRs.txt"
                )

            logger.info(
                f"Found *{len(possible_brs)}* bond "
                f"rearrangement(s) that lead to products"
            )
            return possible_brs

    return None


def save_bond_rearrangs_to_file(brs, filename="bond_rearrangs.txt"):
    """
    Save a list of bond rearrangements to a file in plane text

    ---------------------------------------------------------------------------
    Arguments:
        brs (list(autode.bond_rearrangements.BondRearrangement)):

        filename (str):
    """
    logger.info(f"Saving bond rearrangements to {filename}")

    with open(filename, "w") as file:
        for bond_rearrang in brs:
            print("fbonds", file=file)
            for fbond in bond_rearrang.fbonds:
                print(*fbond, file=file)
            print("bbonds", file=file)
            for bbond in bond_rearrang.bbonds:
                print(*bbond, file=file)
            print("end", file=file)

    return None


def get_bond_rearrangs_from_file(filename="bond_rearrangs.txt"):
    """
    Extract a list of bond rearrangements from a file

    ---------------------------------------------------------------------------
    Keyword Arguments:
        filename (str):

    Returns:
        (list(autode.bond_rearrangements.BondRearrangement)):
    """
    logger.info("Getting bond rearrangements from file")

    if not os.path.exists(filename):
        logger.error("No bond rearrangements file")
        return None

    bond_rearrangs = []

    with open(filename, "r") as br_file:
        fbonds_block = False
        fbonds, bbonds = [], []
        for line in br_file:
            if "fbonds" in line:
                fbonds_block = True

            if "bbonds" in line:
                fbonds_block = False

            if len(line.split()) == 2:
                atom_idx0, atom_idx1 = (int(val) for val in line.split())

                if fbonds_block:
                    fbonds.append((atom_idx0, atom_idx1))
                if not fbonds_block:
                    bbonds.append((atom_idx0, atom_idx1))

            if "end" in line:
                bond_rearrangs.append(
                    BondRearrangement(
                        forming_bonds=fbonds, breaking_bonds=bbonds
                    )
                )
                fbonds = []
                bbonds = []

    return bond_rearrangs


def add_bond_rearrangment(bond_rearrangs, reactant, product, fbonds, bbonds):
    """
    For a possible bond rearrangement, sees if the products are made, and
    adds it to the bond rearrang list if it does

    ---------------------------------------------------------------------------
    Arguments:
        bond_rearrangs (list(autode.bond_rearrangements.BondRearrangement)):
                        list of working bond rearrangements

        reactant (Species): Reactant complex

        product (Species): Product complex

        fbonds (list(tuple)): list of bonds to be made

        bbonds (list(tuple)): list of bonds to be broken

    Returns:
        (list(autode.bond_rearrangements.BondRearrangement)):
    """

    rearranged_graph = generate_rearranged_graph(
        reactant.graph, fbonds=fbonds, bbonds=bbonds
    )

    if is_isomorphic(rearranged_graph, product.graph):
        ordered_fbonds = []
        ordered_bbonds = []
        for fbond in fbonds:
            if fbond[0] < fbond[1]:
                ordered_fbonds.append((fbond[0], fbond[1]))
            else:
                ordered_fbonds.append((fbond[1], fbond[0]))
        for bbond in bbonds:
            if bbond[0] < bbond[1]:
                ordered_bbonds.append((bbond[0], bbond[1]))
            else:
                ordered_bbonds.append((bbond[1], bbond[0]))

        ordered_fbonds.sort()
        ordered_bbonds.sort()
        bond_rearrangs.append(
            BondRearrangement(
                forming_bonds=ordered_fbonds, breaking_bonds=ordered_bbonds
            )
        )

    return bond_rearrangs


def generate_rearranged_graph(graph, fbonds, bbonds):
    """Generate a rearranged graph by breaking bonds (edge) and forming others
    (edge)

    ---------------------------------------------------------------------------
    Arguments:
        graph (nx.Graph): reactant graph

        fbonds (list[tuple]): list of bonds to be made

        bbonds (list[tuple]): list of bonds to be broken

    Returns:
        (nx.Graph): rearranged graph
    """

    rearranged_graph = graph.copy()
    for fbond in fbonds:
        rearranged_graph.add_edge(*fbond)
    for bbond in bbonds:
        rearranged_graph.remove_edge(*bbond)

    return rearranged_graph


def strip_equiv_bond_rearrs(possible_brs, mol, depth=6):
    """Remove any bond rearrangement from possible_brs for which
    there is already an equivalent in the unique_bond_rearrangements list

    ---------------------------------------------------------------------------
    Arguments:
        possible_brs (list(BondRearrangement)):
        mol (autode.species.Complex): Reactant

    Keyword Arguments:
        depth (int): Depth of neighbour list that must be identical for a set
               of atoms to be considered equivalent (default: {6})

    Returns:
        (list(BondRearrangement)): stripped list of BondRearrangement objects
    """
    logger.info(
        "Stripping the forming and breaking bond list by discarding "
        "rearrangements with equivalent atoms"
    )

    unique_brs = []

    for br in possible_brs:
        bond_rearrang_is_unique = True

        # Compare bond_rearrang to all those already considered to be unique,
        for unique_br in unique_brs:
            if unique_br.get_active_atom_neighbour_lists(
                species=mol, depth=depth
            ) == br.get_active_atom_neighbour_lists(species=mol, depth=depth):
                bond_rearrang_is_unique = False

        if bond_rearrang_is_unique:
            unique_brs.append(br)

    logger.info(
        f"Stripped {len(possible_brs) - len(unique_brs)} "
        "bond rearrangements"
    )
    return unique_brs


def prune_small_ring_rearrs(possible_brs, mol):
    """
    Remove any bond rearrangements that go via small (3, 4) rings if there is
    an alternative that goes vie

    ---------------------------------------------------------------------------
    Arguments:
        possible_brs (list(BondRearrangement)):

        mol (autode.species.Complex): Reactant
    """
    small_ring_sizes = (3, 4)

    if not Config.skip_small_ring_tss:
        logger.info("Not pruning small ring TSs")
        return None

    # Membered-ness of rings in each bond rearrangement
    n_mem_rings = [br.n_membered_rings(mol) for br in possible_brs]

    # Unique elements involved in each bond rearrangement
    elems = [
        set(
            mol.atoms[i].label
            for i in range(mol.n_atoms)
            if i in br.active_atoms
        )
        for br in possible_brs
    ]

    logger.info(
        f"Pruning {len(possible_brs)} to remove any "
        f"{small_ring_sizes}-membered rings where others are possible"
    )

    excluded_idxs = []
    for i, br in enumerate(possible_brs):
        logger.info(
            f"Checking bond rearrangement {i} with rings:"
            f" {n_mem_rings[i]} and atom indexes: {br}"
        )

        # Only consider brs with at least one small ring
        if not any(n_mem in small_ring_sizes for n_mem in n_mem_rings[i]):
            continue

        # Check against all other rearrangements
        for j, other_br in enumerate(possible_brs):
            # Only consider brs with the same set of elements
            if elems[i] != elems[j]:
                continue

            # Needs to have the same number of rings
            if len(n_mem_rings[i]) != len(n_mem_rings[j]):
                continue

            # Exclude i if j has a larger smallest ring size
            if min(n_mem_rings[i]) < min(n_mem_rings[j]):
                excluded_idxs.append(i)
                break

    logger.info(
        f"Excluding {len(excluded_idxs)} bond rearrangements based on "
        f"small rings"
    )

    # Delete the excluded bond rearrangements (sorted high ->  low, so the
    # idxs remain the same while deleting)
    for idx in sorted(excluded_idxs, reverse=True):
        del possible_brs[idx]

    return None


class BondRearrangement:
    def __eq__(self, other):
        return self.fbonds == other.fbonds and self.bbonds == other.bbonds

    def __str__(self):
        return "_".join(f"{bond[0]}-{bond[1]}" for bond in self.all)

    def __repr__(self):
        br_str = "Form(" + ",".join(
            f"{bond[0]}-{bond[1]}" for bond in self.fbonds
        )
        br_str += ")+Break(" + ",".join(
            f"{bond[0]}-{bond[1]}" for bond in self.bbonds
        )
        return br_str + ")"

    def get_active_atom_neighbour_lists(self, species, depth):
        """
        Get neighbour lists of all the active atoms in the molecule
        (reactant complex)

        -----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species | autode.species.Complex):
            depth (int): Depth of the neighbour list to consider

        Returns:
            (list(list(str))):
        """

        def nl(idx):
            mol_idxs = None

            try:
                mol_idxs = next(
                    species.atom_indexes(i)
                    for i in range(species.n_molecules)
                    if idx in species.atom_indexes(i)
                )

            except (StopIteration, AttributeError):
                logger.warning("Active atom index not found in any molecules")

            nl_labels = get_neighbour_list(
                species, atom_i=idx, index_set=mol_idxs
            )
            return nl_labels[:depth]

        return [nl(idx) for idx in self.active_atoms]

    def n_membered_rings(self, mol):
        """
        Find the membered-ness of the rings involved in this bond rearrangement
        will add the forming bonds to the graph to determine

        -----------------------------------------------------------------------
        Arguments:
            (autode.species.Species):

        Returns:
            (list(int)):
        """
        assert mol.graph is not None
        graph = mol.graph.copy()

        for fbond in self.fbonds:
            if fbond not in graph.edges:
                graph.add_edge(*fbond)

        rings = find_cycles(graph)
        n_mem_rings = []

        # Full enumeration over all atoms and rings - could be faster..
        for ring in rings:
            for atom_idx in self.active_atoms:
                if atom_idx in ring:
                    # This ring has at least one active atom in
                    n_mem_rings.append(len(ring))

                    # don't add the same ring more than once
                    break

        return n_mem_rings

    @property
    def fatoms(self):
        """Unique atoms indexes involved in forming bonds"""
        return list(sorted(set([i for bond in self.fbonds for i in bond])))

    @property
    def batoms(self):
        """Unique atoms indexes involved in breaking bonds"""
        return list(sorted(set([i for bond in self.bbonds for i in bond])))

    @property
    def active_atoms(self):
        """Unique atom indexes in forming or breaking bonds"""
        return list(sorted(set(a for b in self.all for a in b)))

    @property
    def n_fbonds(self):
        return len(self.fbonds)

    @property
    def n_bbonds(self):
        return len(self.bbonds)

    def __init__(
        self,
        forming_bonds: Optional[list[tuple]] = None,
        breaking_bonds: Optional[list[tuple]] = None,
    ):
        """
        Bond rearrangement

        -----------------------------------------------------------------------
        Keyword Arguments:
            forming_bonds (list(tuple(int))): List of atom pairs that are
                        forming in this reaction

            breaking_bonds (list(tuple(int))): List of atom pairs that are
                           breaking in the reaction
        """

        self.fbonds = forming_bonds if forming_bonds is not None else []
        self.bbonds = breaking_bonds if breaking_bonds is not None else []

        self.all = self.fbonds + self.bbonds
