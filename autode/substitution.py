import numpy as np
from autode.log import logger
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix


def set_complex_xyzs_translated_rotated(reac_complex, reactants, bond_rearrangement, shift_factor=2):
    logger.info('Translating reactant atoms into reactive complex')
    reac_complex_coords = reac_complex.get_coords()

    all_attacked_atoms = get_attacked_atom(bond_rearrangement)
    all_attacked_atom_coords = [reac_complex_coords[attacked_atom]
                                for attacked_atom in all_attacked_atoms]
    avg_attacked_atom_coords = np.average(all_attacked_atom_coords, axis=0)
    # The second reactant has been shifted in the formation of the reactant complex, so move it back such that the
    # atom in the second
    reac_complex_coords = reac_complex_coords - avg_attacked_atom_coords

    if (all(all_attacked_atoms) < reactants[0].n_atoms) or (all(all_attacked_atoms) >= reactants[0].n_atoms):
        atoms_to_shift = (range(reactants[0].n_atoms, reac_complex.n_atoms) if all_attacked_atoms[0] < reactants[0].n_atoms
                          else range(reactants[0].n_atoms))

    else:
        logger.critical(
            'Attacked atoms in both molecules not currently supported')
        exit()

    normed_lg_vector = get_normalised_lg_vector(
        bond_rearrangement, all_attacked_atoms, reac_complex_coords)

    all_fr_atoms = [get_lg_or_fr_atom(bond_rearrangement.fbonds, attacked_atom)
                    for attacked_atom in all_attacked_atoms]
    all_fr_coords = [reac_complex_coords[fr_atom].copy()
                     for fr_atom in all_fr_atoms]
    avg_fr_coords = np.average(all_fr_coords, axis=0)

    # Shift the forming bond atom onto the attacked atoms, then shift along the normalised lg_vector
    for i in atoms_to_shift:
        reac_complex_coords[i] -= avg_fr_coords

    # Get the vector of attack of the fragment with the forming bond
    if all([reac.n_atoms > 1 for reac in reactants]):
        logger.info('Rotating into best attack')
        normed_attack_vector = get_normalised_attack_vector(
            reac_complex, reac_complex_coords, all_fr_atoms)

        rot_matrix = get_rot_matrix(normed_attack_vector, normed_lg_vector)

        for i in atoms_to_shift:
            reac_complex_coords[i] = np.matmul(
                rot_matrix, reac_complex_coords[i])
            reac_complex_coords[i] += shift_factor * normed_lg_vector

    else:
        logger.info('Only had a single atom to shift, will skip rotation')
        for i in atoms_to_shift:
            reac_complex_coords[i] += shift_factor * normed_lg_vector

    reac_complex_xyzs = coords2xyzs(
        coords=reac_complex_coords, old_xyzs=reac_complex.xyzs)
    return reac_complex.set_xyzs(reac_complex_xyzs)


def get_normalised_lg_vector(bond_rearrangement, all_attacked_atoms, reac_complex_coords, tolerance=0.09):
    """
    Get the vector from the attacked atom to the one bonded to tbe breaking bond

    :param bond_rearrangement: (object)
    :param attacked_atom: (int)
    :param reac_complex_coords: (list(nd.array))
    :return: (np.array) normalised vector
    """
    all_lg_vectors = []
    all_lg_atoms = []
    all_attacked_atom_coords = []
    for attacked_atom in all_attacked_atoms:
        attacked_atom_coords = reac_complex_coords[attacked_atom].copy()
        all_attacked_atom_coords.append(attacked_atom_coords)
        lg_atom = get_lg_or_fr_atom(bond_rearrangement.bbonds, attacked_atom)
        all_lg_atoms.append(lg_atom)
        lg_vector = attacked_atom_coords - reac_complex_coords[lg_atom]
        all_lg_vectors.append(lg_vector / np.linalg.norm(lg_vector))
    if len(all_lg_atoms) == 1:
        lg_vector = all_lg_vectors[0]
    else:
        theta = np.arccos(np.dot(all_lg_vectors[0], all_lg_vectors[1]))
        if theta > np.pi - tolerance:
            vector1 = all_lg_vectors[1]
            vector2 = reac_complex_coords[all_lg_atoms[0]
                                          ] - attacked_atom_coords[1]
            lg_vector = np.cross(vector1, vector2)
        else:
            lg_vector = np.average(all_lg_vectors, axis=0)
    return lg_vector / np.linalg.norm(lg_vector)


def get_normalised_attack_vector(reac_complex, reac_complex_coords, fr_atoms, tolerance=0.09):
    all_attack_vectors = []
    for fr_atom in fr_atoms:
        fr_coords = reac_complex_coords[fr_atom].copy()
        fr_bonded_atoms = reac_complex.get_bonded_atoms_to_i(atom_i=fr_atom)
        fr_bond_vectors = [fr_coords - reac_complex_coords[i]
                           for i in range(reac_complex.n_atoms) if i in fr_bonded_atoms]
        if len(fr_bonded_atoms) < 3:
            avg_attack = np.average(fr_bond_vectors, axis=0)
            normed_avg_attack = avg_attack / np.linalg.norm(avg_attack)
            all_attack_vectors.append([normed_avg_attack, False])
        else:
            # check if it is flat, if so want perp vector to plane, if not can take average of the bonds
            logger.info(
                f'Checking if attacking atom ({fr_atom}) has flat geometry')
            flat = True
            for bonded_atom in fr_bonded_atoms:
                for second_bonded_atom in fr_bonded_atoms:
                    if bonded_atom != second_bonded_atom:
                        for third_bonded_atom in fr_bonded_atoms:
                            if third_bonded_atom not in (bonded_atom, second_bonded_atom):
                                bond_vector = fr_coords - \
                                    reac_complex_coords[bonded_atom]
                                second_bond_vector = fr_coords - \
                                    reac_complex_coords[second_bonded_atom]
                                third_bond_vector = fr_coords - \
                                    reac_complex_coords[third_bonded_atom]
                                first_cross_product = np.cross(
                                    bond_vector, second_bond_vector)
                                second_cross_product = np.cross(
                                    bond_vector, third_bond_vector)
                                normed_first_cross_product = first_cross_product / \
                                    np.linalg.norm(first_cross_product)
                                normed_second_cross_product = second_cross_product / \
                                    np.linalg.norm(second_cross_product)
                                theta = np.arccos(
                                    np.dot(normed_first_cross_product, normed_second_cross_product))
                                if tolerance < theta < (np.pi - tolerance):
                                    flat = False

            if flat:
                logger.info(
                    f'Attacking atom ({fr_atom}) has flat geometry')
                all_attack_vectors.append([normed_first_cross_product, True])
            else:
                logger.info('Attacking atom does not have flat geometry')
                avg_attack = np.average(fr_bond_vectors, axis=0)
                normed_avg_attack = avg_attack / np.linalg.norm(avg_attack)
                all_attack_vectors.append([normed_avg_attack, False])
    logger.info('Getting average attack vector')
    # since the flat atom can attack from either side, check which is best with other attacking atom
    if len(all_attack_vectors) == 1:
        best_avg_attack_vector = all_attack_vectors[0][0]
    else:
        if all_attack_vectors[0][1]:
            theta1 = np.arccos(
                np.dot(all_attack_vectors[0][0], all_attack_vectors[1][0]))
            opposite_attack = np.negative(all_attack_vectors[0][0])
            theta2 = np.arccos(
                np.dot(opposite_attack, all_attack_vectors[1][0]))
            if theta2 < theta1:
                all_attack_vectors[0][0] = opposite_attack
        if all_attack_vectors[1][1]:
            theta1 = np.arccos(
                np.dot(all_attack_vectors[0][0], all_attack_vectors[1][0]))
            opposite_attack = np.negative(all_attack_vectors[1][0])
            theta2 = np.arccos(
                np.dot(all_attack_vectors[0][0], opposite_attack))
            if theta2 < theta1:
                all_attack_vectors[1][0] = opposite_attack
        best_attack_vectors = [all_attack_vectors[0]
                               [0], all_attack_vectors[1][0]]
        best_avg_attack_vector = np.average(best_attack_vectors, axis=0)
    if np.linalg.norm(best_avg_attack_vector) == 0:
        best_avg_attack_vector = all_attack_vectors[0][0]
    return best_avg_attack_vector / np.linalg.norm(best_avg_attack_vector)


def get_rot_matrix(normed_attack_vector, normed_lg_vector):

    theta = np.pi - np.arccos(np.dot(normed_attack_vector, normed_lg_vector))
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
            [possible_attacked_atoms.append(atom)
             for atom in fbond if atom in bbond]

    if len(possible_attacked_atoms) > 1:
        logger.warning('Multiple possible attacked atoms in reaction {}'.format(
            possible_attacked_atoms))
        possible_attacking_atoms = []
        if len(possible_attacked_atoms) > 2:
            logger.critical('More than 2 attacked atoms not supported')
            exit()
        for attacked_atom in possible_attacked_atoms:
            attacking_atom = get_lg_or_fr_atom(
                bond_rearrangement.fbonds, attacked_atom)
            possible_attacking_atoms.append(attacking_atom)
        if len(possible_attacked_atoms) == len(possible_attacking_atoms):
            return possible_attacked_atoms
        else:
            logger.critical(
                f'Different number of attacking atoms ({possible_attacking_atoms}) and attacked atoms ({possible_attacked_atoms})')
            exit()
    else:
        return possible_attacked_atoms


def get_lg_or_fr_atom(bbonds_or_fbonds, attacked_atom):
    """
    Get the atom that attaches the attacked atom to the rest of the molecule. Each attacked
    atom is only attacked by one atom, so returning immediately after finding one is fine

    :param bbonds_or_fbonds: (list(tuple)) List of either breaking or forming bonds
    :param attacked_atom: (int) index of the attacked atom in the structure
    :return:
    """

    for bond in bbonds_or_fbonds:
        if bond[0] == attacked_atom:
            return bond[1]
        if bond[1] == attacked_atom:
            return bond[0]
    logger.critical('Couldn\'t find a leaving group atom')
    exit()
