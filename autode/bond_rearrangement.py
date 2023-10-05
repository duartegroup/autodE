import itertools
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

    possible_brs = []

    reac_bond_dict = get_bond_type_list(reactant.graph)
    prod_bond_dict = get_bond_type_list(product.graph)

    # list of bonds where this type of bond (e.g C-H) has less bonds in
    # products than reactants
    all_possible_bbonds = []

    # list of bonds that can be formed of this bond type. This is only used
    # if there is only one type of bbond, so can be overwritten for each new
    # type of bbond
    bbond_atom_type_fbonds = None

    # list of bonds where this type of bond (e.g C-H) has more bonds in
    #  products than reactants
    all_possible_fbonds = []

    # list of bonds that can be broken of this bond type. This is only used
    # if there is only one type of fbond, so can be overwritten for each new
    # type of fbond
    fbond_atom_type_bbonds = None

    # list of bonds where this type of bond (e.g C-H) has the same number of
    # bonds in products and reactants
    possible_bbond_and_fbonds = []

    for reac_key, reac_bonds in reac_bond_dict.items():
        prod_bonds = prod_bond_dict[reac_key]
        possible_fbonds = get_fbonds(reactant.graph, reac_key)
        if len(prod_bonds) < len(reac_bonds):
            all_possible_bbonds.append(reac_bonds)
            bbond_atom_type_fbonds = possible_fbonds
        elif len(prod_bonds) > len(reac_bonds):
            all_possible_fbonds.append(possible_fbonds)
            fbond_atom_type_bbonds = reac_bonds
        else:
            if len(reac_bonds) != 0:
                possible_bbond_and_fbonds.append([reac_bonds, possible_fbonds])

    # The change in the number of bonds is > 0 as in the reaction
    # initialisation reacs/prods are swapped if this is < 0
    delta_n_bonds = (
        reactant.graph.number_of_edges() - product.graph.number_of_edges()
    )

    if delta_n_bonds == 0:
        funcs = [get_fbonds_bbonds_1b1f, get_fbonds_bbonds_2b2f]
    elif delta_n_bonds == 1:
        funcs = [get_fbonds_bbonds_1b, get_fbonds_bbonds_2b1f]
    elif delta_n_bonds == 2:
        funcs = [get_fbonds_bbonds_2b]
    else:
        logger.error(
            f"Cannot treat a change in bonds "
            f"reactant <- product of {delta_n_bonds}"
        )
        return None

    for func in funcs:
        possible_brs = func(
            reactant,
            product,
            possible_brs,
            all_possible_bbonds,
            all_possible_fbonds,
            possible_bbond_and_fbonds,
            bbond_atom_type_fbonds,
            fbond_atom_type_bbonds,
        )

        if len(possible_brs) > 0:
            logger.info(
                f"Found a molecular graph rearrangement to products "
                f"with {func.__name__}"
            )
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

        reactant (autode.species.Complex): Reactant complex

        product (autode.species.Complex): Product complex

        fbonds (list(tuple)): list of bonds to be made

        bbonds (list(tuple)): list of bonds to be broken

    Returns:
        (list(autode.bond_rearrangements.BondRearrangement)):
    """

    # Check that the bond rearrangement doesn't exceed standard atom valances
    bbond_atoms = [atom for bbond in bbonds for atom in bbond]
    for fbond in fbonds:
        for idx in fbond:
            if (
                reactant.graph.degree(idx)
                == reactant.atoms[idx].maximal_valance
                and idx not in bbond_atoms
            ):
                # If we are here then there is at least one atom that will
                # exceed it's maximal valance, therefore
                # we don't need to run isomorphism
                return bond_rearrangs

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

        fbonds (list(tuple)): list of bonds to be made

        bbonds (list(tuple)): list of bonds to be broken

    Returns:
        nx.Graph: rearranged graph
    """

    rearranged_graph = graph.copy()
    for fbond in fbonds:
        rearranged_graph.add_edge(*fbond)
    for bbond in bbonds:
        rearranged_graph.remove_edge(*bbond)

    return rearranged_graph


def get_fbonds_bbonds_1b(
    reac,
    prod,
    possible_brs,
    all_possible_bbonds,
    all_possible_fbonds,
    possible_bbond_and_fbonds,
    bbond_atom_type_fbonds,
    fbond_atom_type_bbonds,
):
    logger.info("Getting possible 1 breaking bond rearrangements")

    for bbond in all_possible_bbonds[0]:
        # Break one bond
        possible_brs = add_bond_rearrangment(
            possible_brs, reac, prod, fbonds=[], bbonds=[bbond]
        )

    return possible_brs


def get_fbonds_bbonds_2b(
    reac,
    prod,
    possible_brs,
    all_possible_bbonds,
    all_possible_fbonds,
    possible_bbond_and_fbonds,
    bbond_atom_type_fbonds,
    fbond_atom_type_bbonds,
):
    logger.info("Getting possible 2 breaking bond rearrangements")

    if len(all_possible_bbonds) == 1:
        # Break two bonds of the same type
        for bbond1, bbond2 in itertools.combinations(
            all_possible_bbonds[0], 2
        ):
            possible_brs = add_bond_rearrangment(
                possible_brs, reac, prod, fbonds=[], bbonds=[bbond1, bbond2]
            )

    elif len(all_possible_bbonds) == 2:
        # Break two bonds of different types
        for bbond1, bbond2 in itertools.product(
            all_possible_bbonds[0], all_possible_bbonds[1]
        ):
            possible_brs = add_bond_rearrangment(
                possible_brs, reac, prod, fbonds=[], bbonds=[bbond1, bbond2]
            )

    return possible_brs


def get_fbonds_bbonds_1b1f(
    reac,
    prod,
    possible_brs,
    all_possible_bbonds,
    all_possible_fbonds,
    possible_bbond_and_fbonds,
    bbond_atom_type_fbonds,
    fbond_atom_type_bbonds,
):
    logger.info(
        "Getting possible 1 breaking and 1 forming bond " "rearrangements"
    )

    if len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        # Make and break a bond of different types
        for fbond, bbond in itertools.product(
            all_possible_fbonds[0], all_possible_bbonds[0]
        ):
            possible_brs = add_bond_rearrangment(
                possible_brs, reac, prod, fbonds=[fbond], bbonds=[bbond]
            )

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 0:
        # Make and break a bond of the same type
        for bbonds, fbonds in possible_bbond_and_fbonds:
            for bbond, fbond in itertools.product(bbonds, fbonds):
                possible_brs = add_bond_rearrangment(
                    possible_brs, reac, prod, fbonds=[fbond], bbonds=[bbond]
                )

    return possible_brs


def get_fbonds_bbonds_2b1f(
    reac,
    prod,
    possible_brs,
    all_possible_bbonds,
    all_possible_fbonds,
    possible_bbond_and_fbonds,
    bbond_atom_type_fbonds,
    fbond_atom_type_bbonds,
):
    logger.info(
        "Getting possible 2 breaking and 1 forming bond rearrangements"
    )

    if len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 1:
        # Make a bond and break two bonds, all of different types
        possibles = itertools.product(
            all_possible_fbonds[0],
            all_possible_bbonds[0],
            all_possible_bbonds[1],
        )

        for fbond, bbond1, bbond2 in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond],
                bbonds=[bbond1, bbond2],
            )

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        # Make a bond of one type, break two bonds of another type
        two_same_possibles = itertools.combinations(all_possible_bbonds[0], 2)
        possibles = itertools.product(
            all_possible_fbonds[0], two_same_possibles
        )

        for fbond, (bbond1, bbond2) in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond],
                bbonds=[bbond1, bbond2],
            )

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 0:
        for bbonds, fbonds in possible_bbond_and_fbonds:
            # Make and break a bond of one type, break a bond of a different
            # type
            possibles = itertools.product(
                fbonds, all_possible_bbonds[0], bbonds
            )

            for fbond, bbond1, bbond2 in possibles:
                possible_brs = add_bond_rearrangment(
                    possible_brs,
                    reac,
                    prod,
                    fbonds=[fbond],
                    bbonds=[bbond1, bbond2],
                )

        # Make and break two bonds, all of the same type
        two_same_possibles = itertools.combinations(all_possible_bbonds[0], 2)
        possibles = itertools.product(
            bbond_atom_type_fbonds, two_same_possibles
        )

        for fbond, (bbond1, bbond2) in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond],
                bbonds=[bbond1, bbond2],
            )

    return possible_brs


def get_fbonds_bbonds_2b2f(
    reac,
    prod,
    possible_brs,
    all_possible_bbonds,
    all_possible_fbonds,
    possible_bbond_and_fbonds,
    bbond_atom_type_fbonds,
    fbond_atom_type_bbonds,
):
    logger.info(
        "Getting possible 2 breaking and 2 forming bond rearrangements"
    )

    if len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 2:
        # Make two bonds and break two bonds, all of different types
        possibles = itertools.product(
            all_possible_fbonds[0],
            all_possible_fbonds[1],
            all_possible_bbonds[0],
            all_possible_bbonds[1],
        )

        for fbond1, fbond2, bbond1, bbond2 in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond1, fbond2],
                bbonds=[bbond1, bbond2],
            )

    elif len(all_possible_bbonds) == 2 and len(all_possible_fbonds) == 1:
        # Make two bonds of the same type, break two bonds of different types
        two_same_possibles = itertools.combinations(all_possible_fbonds[0], 2)
        possibles = itertools.product(
            all_possible_bbonds[0], all_possible_bbonds[1], two_same_possibles
        )

        for bbond1, bbond2, (fbond1, fbond2) in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond1, fbond2],
                bbonds=[bbond1, bbond2],
            )

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 2:
        # Make two bonds of different types, break two bonds of the same type
        two_same_possibles = itertools.combinations(all_possible_bbonds[0], 2)
        possibles = itertools.product(
            all_possible_fbonds[0], all_possible_fbonds[1], two_same_possibles
        )

        for fbond1, fbond2, (bbond1, bbond2) in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond1, fbond2],
                bbonds=[bbond1, bbond2],
            )

    elif len(all_possible_bbonds) == 1 and len(all_possible_fbonds) == 1:
        two_f_possibles = itertools.combinations(all_possible_fbonds[0], 2)
        two_b_possibles = itertools.combinations(all_possible_bbonds[0], 2)
        possibles = itertools.product(two_f_possibles, two_b_possibles)

        for (fbond1, fbond2), (bbond1, bbond2) in possibles:
            # Make two bonds of the same type, break two bonds of another type
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond1, fbond2],
                bbonds=[bbond1, bbond2],
            )

        for bbonds, fbonds in possible_bbond_and_fbonds:
            # Make one bonds of one type, break one bond of another type, make
            # and break a bond of a third type
            possibles = itertools.product(
                all_possible_fbonds[0], fbonds, all_possible_bbonds[0], bbonds
            )

            for fbond1, fbond2, bbond1, bbond2 in possibles:
                possible_brs = add_bond_rearrangment(
                    possible_brs,
                    reac,
                    prod,
                    fbonds=[fbond1, fbond2],
                    bbonds=[bbond1, bbond2],
                )

        # Make a bond of one type, make and break two bonds of another type
        two_b_possibles = itertools.combinations(all_possible_bbonds[0], 2)
        possibles = itertools.product(
            all_possible_fbonds[0], bbond_atom_type_fbonds, two_b_possibles
        )

        for fbond1, fbond2, (bbond1, bbond2) in possibles:
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond1, fbond2],
                bbonds=[bbond1, bbond2],
            )

        two_f_possibles = itertools.combinations(all_possible_fbonds[0], 2)
        possibles = itertools.product(
            all_possible_bbonds[0], fbond_atom_type_bbonds, two_f_possibles
        )

        for bbond1, bbond2, (fbond1, fbond2) in possibles:
            # Break a bond of one type, make two and break one bond of another
            #  type
            possible_brs = add_bond_rearrangment(
                possible_brs,
                reac,
                prod,
                fbonds=[fbond1, fbond2],
                bbonds=[bbond1, bbond2],
            )

    elif len(all_possible_bbonds) == 0 and len(all_possible_fbonds) == 0:
        possibles_b_f = itertools.combinations(possible_bbond_and_fbonds, 2)

        for (bbonds1, fbonds1), (bbonds2, fbonds2) in possibles_b_f:
            # Make and break a bond of one type, make and break a bond of
            # another type
            possibles = itertools.product(fbonds1, bbonds1, fbonds2, bbonds2)

            for fbond1, bbond1, fbond2, bbond2 in possibles:
                possible_brs = add_bond_rearrangment(
                    possible_brs,
                    reac,
                    prod,
                    fbonds=[fbond1, fbond2],
                    bbonds=[bbond1, bbond2],
                )

        for bbonds, fbonds in possible_bbond_and_fbonds:
            # Make two and break two bonds, all of the same type
            possibles = itertools.product(
                itertools.combinations(fbonds, 2),
                itertools.combinations(bbonds, 2),
            )

            for (fbond1, fbond2), (bbond1, bbond2) in possibles:
                possible_brs = add_bond_rearrangment(
                    possible_brs,
                    reac,
                    prod,
                    fbonds=[fbond1, fbond2],
                    bbonds=[bbond1, bbond2],
                )

    return possible_brs


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

    def __init__(self, forming_bonds=None, breaking_bonds=None):
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
