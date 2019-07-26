import numpy as np
from .log import logger
from .geom import coords2xyzs
from .geom import calc_rotation_matrix


def set_complex_xyzs_translated_rotated(reac_complex, reactants, bond_rearrangement, shift_factor=2.0):
    logger.info('Translating reactant atoms into reactive complex')
    reac_complex_coords = reac_complex.get_coords()

    attacked_atom = get_attacked_atom(bond_rearrangement)
    attacked_atom_coords = reac_complex_coords[attacked_atom]
    # The second reactant has been shifted in the formation of the reactant complex, so move it back such that the
    # atom in the second
    reac_complex_coords = reac_complex_coords - attacked_atom_coords

    atoms_to_shift = (range(reactants[0].n_atoms, reac_complex.n_atoms) if attacked_atom < reactants[0].n_atoms
                      else range(reactants[0].n_atoms))

    normed_lg_vector = get_normalised_lg_vector(bond_rearrangement, attacked_atom, reac_complex_coords)

    fr_atom = get_lg_or_fr_atom(bond_rearrangement.fbonds, attacked_atom)
    fr_coords = reac_complex_coords[fr_atom].copy()

    # Shift the forming bond atom onto the attacked atoms, then shift along the normalised lg_vector
    for i in atoms_to_shift:
        reac_complex_coords[i] -= fr_coords

    # Get the vector of attack of the fragment with the forming bond
    if all([reac.n_atoms > 1 for reac in reactants]):
        logger.info('Rotating into best 180 degree attack')
        normed_attack_vector = get_normalised_attack_vetor(reac_complex, reac_complex_coords, fr_atom)
        rot_matrix = get_rot_matrix(normed_attack_vector, normed_lg_vector)

        for i in atoms_to_shift:
            reac_complex_coords[i] = np.matmul(rot_matrix, reac_complex_coords[i])
            reac_complex_coords[i] += shift_factor * normed_lg_vector
    else:
        logger.info('Only had a single atom to shift, will skip rotation')
        for i in atoms_to_shift:
            reac_complex_coords[i] += shift_factor * normed_lg_vector

    reac_complex_xyzs = coords2xyzs(coords=reac_complex_coords, old_xyzs=reac_complex.xyzs)
    return reac_complex.set_xyzs(reac_complex_xyzs)


def get_normalised_lg_vector(bond_rearrangement, attacked_atom, reac_complex_coords):
    """
    Get the vector from the attacked atom to the one bonded to tbe breaking bond

    :param bond_rearrangement: (object)
    :param attacked_atom: (int)
    :param reac_complex_coords: (list(nd.array))
    :return: (np.array) normalised vector
    """
    attacked_atom_coords = reac_complex_coords[attacked_atom].copy()
    lg_atom = get_lg_or_fr_atom(bond_rearrangement.bbonds, attacked_atom)
    lg_vector = attacked_atom_coords - reac_complex_coords[lg_atom]

    return lg_vector / np.linalg.norm(lg_vector)


def get_normalised_attack_vetor(reac_complex, reac_complex_coords, fr_atom):
    fr_coords = reac_complex_coords[fr_atom].copy()
    fr_bonded_atoms = reac_complex.get_bonded_atoms_to_i(atom_i=fr_atom)
    attack_vetor = np.average([fr_coords - reac_complex_coords[i] for i in range(reac_complex.n_atoms) if i in fr_bonded_atoms], axis=0)

    return attack_vetor / np.linalg.norm(attack_vetor)


def get_rot_matrix(normed_attack_vector, normed_lg_vector):

    theta = np.arccos(np.dot(normed_attack_vector, normed_lg_vector))
    axis = np.cross(normed_lg_vector, normed_attack_vector)
    return calc_rotation_matrix(axis=axis/np.linalg.norm(axis), theta=theta)


def get_attacked_atom(bond_rearrangement):
    """
    For a bond rearrangement find the atom which is common to both a forming and breaking bond in a substitution
    reaction thus has been attacked. Exit if there are multiple
    :param bond_rearrangement: (object)
    :return: (int) index of the attacked atom
    """

    possible_attacked_atoms = []
    for fbond in bond_rearrangement.fbonds:
        for bbond in bond_rearrangement.bbonds:
            [possible_attacked_atoms.append(atom) for atom in fbond if atom in bbond]

    if len(possible_attacked_atoms) > 1:
        logger.critical('Multiple possible attacked atoms in Substitution reaction {}'.format(possible_attacked_atoms))
        exit()
    else:
        return possible_attacked_atoms[0]


def get_lg_or_fr_atom(bbonds_or_fbonds, attacked_atom):
    """
    Get the atom that attaches the attacked atom to the rest of the molecule. From get_attacked_atom there should
    only be one possibility in the structure so returning immediately after finding one is fine

    :param bbonds_or_fbonds: (list(tuple)) List of either breaking or forming bonds
    :param attacked_atom: (int) index of the attacked atom in the structure
    :return:
    """

    for bond in bbonds_or_fbonds:
        if bond[0] == attacked_atom:
            return bond[1]
        if bond[1] == attacked_atom:
            return bond[0]
    else:
        logger.critical('Couldn\'t find a leaving group atom')
        exit()
