from copy import deepcopy
import numpy as np
from numpy.linalg import norm as length
from autode.atoms import DummyAtom
from autode.mol_graphs import connected_components
from autode.log import logger


class SubstitutionCentre:
    def __str__(self):
        return (
            f"a_atom = {self.a_atom}, c_atom = {self.c_atom} "
            f"x_atom = {self.x_atom}, a_atom_nns = {self.a_atom_nn}"
        )

    def set_attack_r0(self, species, shift_factor):
        """Set the ideal distance between a and c atoms in a substitution
        centre"""

        r0 = species.atoms.eqm_bond_distance(self.a_atom, self.c_atom)
        self.r0_ac = shift_factor * r0
        return None

    def __init__(self, a_atom_idx, c_atom_idx, x_atom_idx, a_atom_nn_idxs):
        """
        Substitution centre has the following structure::

              H           H  H
              |           |/
              N-- H       C -- Cl
             /           /
            H           H


        where::

              a_atom = N
              c_atom = C
              x_atom = Cl
              a_atom_nn = H, H, H (bonded to N)

        all given as their atom indexes in a ReactantComplex
        """

        self.a_atom = a_atom_idx
        self.c_atom = c_atom_idx
        self.x_atom = x_atom_idx
        self.a_atom_nn = a_atom_nn_idxs

        self.r0_ac = None


def get_substc_and_add_dummy_atoms(reactant, bond_rearrangement, shift_factor):
    """Get all the substitution centers in a molecule. A substitution centre is
    defined as atom that upon reaction has a bond made and broken
    simultaneously

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.ReactantComplex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):

        shift_factor (float): The multiplier in the ideal A--C distance where
                              A is an attacking atom and C a substitution
                              centre

    Returns:
        (tuple(list(autode.substitution.SubstitutionCentre),
               autode.complex.ReactantComplex)):
    """
    logger.info("Finding substitution centers in the reactant")

    subst_centers = []

    for fbond in bond_rearrangement.fbonds:
        for bbond in bond_rearrangement.bbonds:
            if len(set(fbond).intersection(bbond)) == 0:
                # If there are no common atoms between the forming and
                # breaking bonds continue
                continue

            # The attacked (c) atom is the intersection between the
            # breaking and forming bonds
            c_atom = list(set(fbond).intersection(bbond))[0]

            # The leaving group atom is the other atom in the breaking bond
            x_atom = [
                atom_index for atom_index in bbond if atom_index != c_atom
            ][0]

            # The attacked atom is the other atom in the forming bond
            a_atom = [
                atom_index for atom_index in fbond if atom_index != c_atom
            ][0]

            subst_center = SubstitutionCentre(
                a_atom_idx=a_atom,
                c_atom_idx=c_atom,
                x_atom_idx=x_atom,
                a_atom_nn_idxs=[nn for nn in reactant.graph.neighbors(a_atom)],
            )
            subst_center.set_attack_r0(
                species=reactant, shift_factor=shift_factor
            )

            subst_centers.append(subst_center)

    if len(subst_centers) == 0:
        logger.info("No standard A - C - X substitution centres found")

        if (
            len(bond_rearrangement.bbonds) != 1
            or len(bond_rearrangement.fbonds) != 1
        ):
            raise NotImplementedError

        # Add dummy atoms to the reactant to find e.g. SN2' reactions
        add_dummy_atom(reactant, bond_rearrangement)

        # Once a dummy atom has been found then this function should find the
        # *single* substitution centre
        return get_substc_and_add_dummy_atoms(
            reactant, bond_rearrangement, shift_factor
        )

    if any(atom.label == "D" for atom in reactant.atoms):
        logger.info("Removing dummy X atom from bond rearrangement")

        d_atom_idxs = [
            i for i, atom in enumerate(reactant.atoms) if atom.label == "D"
        ]

        # Reset the breaking bond list with only those not containing the
        # dummy atom indexes
        bbonds = [
            bbond
            for bbond in bond_rearrangement.bbonds
            if len(set(bbond).intersection(d_atom_idxs)) == 0
        ]
        bond_rearrangement.bbonds = bbonds

    logger.info(f"Found {len(subst_centers)} substitution centers")
    return subst_centers


def add_dummy_atom(reactant, bond_rearrangement):
    """
    Add a dummy atom above or below the plane of the reactant as a temporary
    X atom

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.ReactantComplex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
    """
    logger.info("Adding dummy X atom so a substitution center can be found")

    fbond = bond_rearrangement.fbonds[0]
    bbond = bond_rearrangement.bbonds[0]

    components = connected_components(reactant.graph)

    if len(components) != 2:
        raise NotImplementedError("Must have two components for dummy add")

    mol1_idxs, mol2_idxs = components

    # Find the central atom as the atom index that is in the forming bond but
    # also contains all indexes of the breaking bond
    if fbond[0] in mol1_idxs and all(idx in mol2_idxs for idx in bbond):
        c_atom = fbond[1]

    else:
        c_atom = fbond[0]

    # Nearest neighbours to the central atom used to generate the normal
    # along which the dummy atom is placed
    c_atom_nns = list(reactant.graph.neighbors(c_atom))

    if len(c_atom_nns) < 2:
        raise NotImplementedError("Cannot place dummy atom")

    cn1, cn2 = c_atom_nns[:2]
    coords = reactant.coordinates

    # Calculate the normal from the vectors to two of the neighbours
    position = np.cross(
        coords[cn1] - coords[c_atom], coords[cn2] - coords[c_atom]
    )
    position /= length(position)

    # Add the dummy atom to a position on the top/bottom face
    logger.warning("Adding a dummy atom to the set of atoms")
    reactant.atoms.append(DummyAtom(*position))

    # Add the breaking bond to the bond rearrangement temporarily
    bond_rearrangement.bbonds.append([c_atom, len(reactant.atoms) - 1])

    return None


def attack_cost(
    reactant, subst_centres, attacking_mol_idx, a=1.0, b=1.0, c=1.0, d=10.0
):
    """
    Calculate the 'attack cost' for a molecule attacking in e.g. a
    substitution or elimination reaction::

        C = Σ_ac a * (r_ac - r^0_ac)^2  +  Σ_acx b * (1 - cos(θ))  +
                  Σ_acx c*(1 + cos(φ))  +  Σ_ij d/r_ij^4

    where::

        cos(θ) = (v_ann • v_cx / |v_ann||v_cx|)
        cos(φ) = (v_ca • v_cx / |v_ca||v_cx|)

    ---------------------------------------------------------------------------
    Returns:
        (float): Cost
    """
    coords = reactant.coordinates
    cost = 0

    for subst_centre in subst_centres:
        r_ac = reactant.distance(i=subst_centre.a_atom, j=subst_centre.c_atom)

        cost += a * (r_ac - subst_centre.r0_ac) ** 2

        # Attack vector is the average of all the nearest neighbour atoms,
        # unless it is flat
        a_nn_coords = [
            coords[atom_index] - coords[subst_centre.a_atom]
            for atom_index in subst_centre.a_atom_nn
        ]

        if len(a_nn_coords) == 0:
            # The attacking atom has no nearest neighbours thus take the
            # attack vector to be a unit vector
            v_ann = np.array([1.0, 0.0, 0.0])
        else:
            v_ann = -np.average(np.array(a_nn_coords), axis=0)

        if length(v_ann) < 1e-1:
            # Attacking atom is planar. Compute the perpendicular from two
            # nearest neighbours
            v_ann = np.cross(
                coords[subst_centre.a_atom]
                - coords[subst_centre.a_atom_nn[0]],
                coords[subst_centre.a_atom]
                - coords[subst_centre.a_atom_nn[1]],
            )

        v_cx = coords[subst_centre.x_atom] - coords[subst_centre.c_atom]

        # b(1 - cos(θ))
        cost += b * (1 - np.dot(v_ann, v_cx) / (length(v_ann) * length(v_cx)))

        v_ca = coords[subst_centre.a_atom] - coords[subst_centre.c_atom]

        # c(1 + cos(φ))
        cost += c * (1 + np.dot(v_ca, v_cx) / (length(v_ca) * length(v_cx)))

        repulsion = reactant.calc_repulsion(mol_index=attacking_mol_idx)
        cost += d * repulsion

    return cost


def get_cost_rotate_translate(x, reactant, subst_centres, attacking_mol_idx):
    """
    Get the cost for placing an attacking mol given a specified rotation and
    translation

    ---------------------------------------------------------------------------
    Arguments:
        x (np.ndarray): Length 11

        reactant (autode.complex.ReactantComplex):

        subst_centres (list(autode.substitution.SubstitutionCentre)):

        attacking_mol_idx (int): Index of the attacking molecule

    Returns:
        (float):
    """

    moved_reactant = deepcopy(reactant)
    moved_reactant.rotate_mol(
        axis=x[:3], theta=x[3], mol_index=attacking_mol_idx
    )

    moved_reactant.translate_mol(vec=x[4:7], mol_index=attacking_mol_idx)

    moved_reactant.rotate_mol(
        axis=x[7:10], theta=x[10], mol_index=attacking_mol_idx
    )

    return attack_cost(moved_reactant, subst_centres, attacking_mol_idx)
