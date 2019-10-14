import numpy as np
from autode.log import logger
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix


def set_complex_xyzs_translated_rotated(reac_complex, reactants, bond_rearrangement, shift_factor=-2):
    logger.info('Translating reactant atoms into reactive complex')
    reac_complex_coords = reac_complex.get_coords()

    all_attacked_atoms = get_attacked_atom(bond_rearrangement)
    all_attacked_atom_coords = [reac_complex_coords[attacked_atom] for attacked_atom in all_attacked_atoms]
    avg_attacked_atom_coords = np.average(all_attacked_atom_coords, axis=0)
    # The second reactant has been shifted in the formation of the reactant complex, so move it back such that the
    # atom in the second
    reac_complex_coords = reac_complex_coords - avg_attacked_atom_coords

    if (all(all_attacked_atoms) < reactants[0].n_atoms) or (all(all_attacked_atoms) >= reactants[0].n_atoms):
        atoms_to_shift = (range(reactants[0].n_atoms, reac_complex.n_atoms) if all_attacked_atoms[0] < reactants[0].n_atoms 
                            else range(reactants[0].n_atoms)) 

    else:
        logger.critical(f'Attacked atoms in both moleucles not currently supported')
        exit()

    all_normed_lg_vector = [get_normalised_lg_vector(bond_rearrangement, attacked_atom, reac_complex_coords) 
                            for attacked_atom in all_attacked_atoms]
    
    normed_lg_vector = check_vectors_directions(all_normed_lg_vector)

    all_fr_atoms = [get_lg_or_fr_atom(bond_rearrangement.fbonds, attacked_atom) 
                        for attacked_atom in all_attacked_atoms]
    all_fr_coords = [reac_complex_coords[fr_atom].copy() for fr_atom in all_fr_atoms]
    avg_fr_coords = np.average(all_fr_coords, axis=0)

    # Shift the forming bond atom onto the attacked atoms, then shift along the normalised lg_vector
    for i in atoms_to_shift:
        reac_complex_coords[i] -= avg_fr_coords

    # Get the vector of attack of the fragment with the forming bond
    if all([reac.n_atoms > 1 for reac in reactants]):
        logger.info('Rotating into best attack')
        all_normed_attack_vector = [get_normalised_attack_vector(reac_complex, reac_complex_coords, fr_atom) 
                                    for fr_atom in all_fr_atoms]
        
        normed_attack_vector = check_vectors_directions(all_normed_attack_vector)
        print(normed_attack_vector)

        rot_matrix = get_rot_matrix(normed_attack_vector, normed_lg_vector)

        for i in atoms_to_shift:
            reac_complex_coords[i] += shift_factor * normed_attack_vector
            reac_complex_coords[i] = np.matmul(rot_matrix, reac_complex_coords[i])

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


def get_normalised_attack_vector(reac_complex, reac_complex_coords, fr_atom, tolerance=0.09):
    fr_coords = reac_complex_coords[fr_atom].copy()
    fr_bonded_atoms = reac_complex.get_bonded_atoms_to_i(atom_i=fr_atom)
    fr_bond_vectors = [fr_coords - reac_complex_coords[i] for i in range(reac_complex.n_atoms) if i in fr_bonded_atoms]
    if len(fr_bonded_atoms) < 3:
        attack_vector = np.average(fr_bond_vectors, axis=0)
    else:
        #check if it is flat, if so want perp vector to plane, if not can take average of the bonds
        perp_vector_1 = np.cross(fr_bond_vectors[0], fr_bond_vectors[1])
        normed_perp_vector_1 = perp_vector_1 / np.linalg.norm(perp_vector_1)
        perp_vector_2 = np.cross(fr_bond_vectors[0], fr_bond_vectors[2])
        normed_perp_vector_2 = perp_vector_2 / np.linalg.norm(perp_vector_2)
        theta = np.arccos(np.dot(normed_perp_vector_1, normed_perp_vector_2))
        print(theta)

        if (tolerance < theta < (np.pi/2 - tolerance)) or (np.pi/2 + tolerance) < theta < (np.pi - tolerance):
            attack_vector = np.average(fr_bond_vectors, axis=0)
        else:
            attack_vector = perp_vector_1

    return attack_vector / np.linalg.norm(attack_vector)


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
        logger.warning('Multiple possible attacked atoms in reaction {}'.format(possible_attacked_atoms))
        possible_attacking_atoms = []
        for attacked_atom in possible_attacked_atoms:
            attacking_atom = get_lg_or_fr_atom(bond_rearrangement.fbonds, attacked_atom)
            possible_attacking_atoms.append(attacking_atom)
        if len(possible_attacked_atoms) > 2:
            logger.critical('More than 2 attacked atoms not supprted')
            exit()
        if len(possible_attacked_atoms) == len(possible_attacking_atoms):
            return possible_attacked_atoms
        else:
            logger.critical(f'Different number of attacking atoms ({possible_attacking_atoms}) and attacked atoms ({possible_attacked_atoms})')    
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
    else:
        logger.critical('Couldn\'t find a leaving group atom')
        exit()


def check_vectors_directions(vectors):
    """Checks if the vectors point the same way, so the best vector will be their average, or opposite ways, and the best vecotr will be the cross product
    
    Arguments:
        vectors {list of vectors} -- list of normalised vectors to check
    """
    if len(vectors) == 1:
        return vectors[0]
    if len(vectors) == 2:
        theta = np.arccos(np.dot(vectors[0], vectors[1]))
        if theta < np.pi/2:
            return np.average(vectors, axis=0)
        else:
            avg_vector = np.cross(vectors[0], vectors[1])
            return avg_vector / np.linalg.norm(avg_vector)
