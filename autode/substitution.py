import numpy as np
from autode.log import logger
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix
from scipy.optimize import minimize
from scipy.spatial import distance_matrix


def set_complex_xyzs_translated_rotated(reac_complex, prod_complex, reactants, bond_rearrangement, mapping, shift_factor):
    """Sets the xyzs in the complex such that the attacking atoms are pointing at the atom they attack

    Arguments:
        reac_complex (mol obj): reactant complex object
        prod_complex (mol obj): product complex object
        reactants (list): list of reactant mol objects
        bond_rearrangement (bond rearrang obj): the bond rearrangement corresponding to the reaction
        mapping (dict): mapping[reactant indices] = product indices
        shift_factor (int): distance the reacting atoms should be apart
    """
    logger.info('Translating reactant atoms into reactive complex')
    reac_complex_coords = reac_complex.get_coords()

    attacked_atoms = get_attacked_atoms(bond_rearrangement)

    if reactants[0].n_atoms / reac_complex.n_atoms > 0.5:
        atoms_to_shift = range(reactants[0].n_atoms, reac_complex.n_atoms)
    else:
        atoms_to_shift = range(reactants[0].n_atoms)

    mol1_vectors = []
    mol2_vectors = []
    mol1_atoms = []
    mol2_atoms = []
    for atom in attacked_atoms:
        if [atom] in mol1_atoms or [atom] in mol2_atoms:
            continue
        bond_breaking_vector = get_normalised_bond_breaking_vector(bond_rearrangement, atom, reac_complex_coords)
        attacking_atoms = (get_leaving_or_attacking_atoms(bond_rearrangement.fbonds, atom))
        attack_vector = get_normalised_attack_vector(reac_complex, reac_complex_coords, attacking_atoms)
        if atom in atoms_to_shift:
            mol1_vectors.append(attack_vector)
            mol2_vectors.append(bond_breaking_vector)
            mol1_atoms.append(attacking_atoms)
            mol2_atoms.append([atom])
        else:
            mol1_vectors.append(bond_breaking_vector)
            mol2_vectors.append(attack_vector)
            mol1_atoms.append([atom])
            mol2_atoms.append(attacking_atoms)

    min_dot = 9999999.9
    best_theta, best_phi, best_z = None, None, None

    for _ in range(100):
        x0 = np.random.uniform(0.0, 2*np.pi, size=3)
        res = minimize(dot_func, x0=x0, args=(mol1_vectors, mol2_vectors), method='BFGS')
        if res.fun < min_dot:
            min_dot = res.fun
            best_theta, best_phi, best_z = res.x

    theta_rot_matrix = calc_rotation_matrix([1, 0, 0], best_theta)
    phi_rot_matrix = calc_rotation_matrix([0, 1, 0], best_phi)
    z_rot_matrix = calc_rotation_matrix([0, 0, 1], best_z)
    rot_matrix = np.matmul(theta_rot_matrix, np.matmul(phi_rot_matrix, z_rot_matrix))

    for i in atoms_to_shift:
        reac_complex_coords[i] = np.matmul(rot_matrix, reac_complex_coords[i])

    mol1_atom_coords = []
    for item in mol1_atoms:
        coords = []
        for atom in item:
            coords.append(reac_complex_coords[atom])
        mol1_atom_coords.append(np.average(coords, axis=0))

    avg_mol1_atom_coord = np.average(mol1_atom_coords, axis=0)
    reac_complex_coords -= avg_mol1_atom_coord

    mol2_atom_coords = []
    for item in mol2_atoms:
        coords = []
        for atom in item:
            coords.append(reac_complex_coords[atom])
        mol2_atom_coords.append(np.average(coords, axis=0))

    avg_mol2_atom_coord = np.average(mol2_atom_coords, axis=0)
    avg_mol1_vector = np.average(mol1_vectors, axis=0)

    for i in atoms_to_shift:
        reac_complex_coords[i] -= avg_mol2_atom_coord - shift_factor * avg_mol1_vector

    if len(mol1_vectors) == 1:
        min_dist = 9999999.9
        best_theta = None
        mol1_coords = reac_complex_coords[[i for i in range(reac_complex.n_atoms) if not i in atoms_to_shift]]
        mol2_coords = reac_complex_coords[atoms_to_shift]
        for _ in range(100):
            x0 = np.random.uniform(0.0, 2*np.pi, size=1)
            res = minimize(rot_func, x0=x0, args=(avg_mol1_vector, mol1_coords, mol2_coords), method='BFGS')
            if res.fun < min_dist:
                min_dist = res.fun
                best_theta = res.x[0]

        rot_matrix = calc_rotation_matrix(avg_mol1_vector, best_theta)
        for i in atoms_to_shift:
            reac_complex_coords[i] = np.matmul(rot_matrix, reac_complex_coords[i])

    reac_complex_xyzs = coords2xyzs(coords=reac_complex_coords, old_xyzs=reac_complex.xyzs)
    reac_complex.set_xyzs(reac_complex_xyzs)


def get_normalised_bond_breaking_vector(bond_rearrangement, attacked_atom, reac_complex_coords):
    """Get the vector from the attacked atom to the one bonded to tbe breaking bond

    Arguments:
        bond_rearrangement (object): bond rearrangement object
        attacked_atom (int): index of attack atom
        reac_complex_coords (np.ndarray): coords of the reac complex

    Returns:
        np.ndarray: avg leaving group vector
    """

    attacked_atom_coords = reac_complex_coords[attacked_atom]
    leaving_atoms = get_leaving_or_attacking_atoms(bond_rearrangement.bbonds, attacked_atom)
    leaving_vectors = [attacked_atom_coords - reac_complex_coords[leaving_atom] for leaving_atom in leaving_atoms]
    leaving_vector = np.average(leaving_vectors, axis=0)
    return leaving_vector / np.linalg.norm(leaving_vector)


def get_normalised_attack_vector(reac_complex, reac_complex_coords, attacking_atoms, tolerance=0.09):
    all_attack_vectors = []
    for attacking_atom in attacking_atoms:
        attacking_coords = reac_complex_coords[attacking_atom]
        attacking_bonded_atoms = reac_complex.get_bonded_atoms_to_i(atom_i=attacking_atom)
        attacking_bond_vectors = [attacking_coords - reac_complex_coords[i] for i in range(reac_complex.n_atoms) if i in attacking_bonded_atoms]
        if len(attacking_bonded_atoms) == 0:
            all_attack_vectors.append([[1, 0, 0], False])
        elif len(attacking_bonded_atoms) < 3:
            avg_attack = np.average(attacking_bond_vectors, axis=0)
            normed_avg_attack = avg_attack / np.linalg.norm(avg_attack)
            all_attack_vectors.append([normed_avg_attack, False])
        else:
            # check if it is flat, if so want perp vector to plane, if not can take average of the bonds
            logger.info(f'Checking if attacking atom ({attacking_atom}) has flat geometry')
            flat = True
            for bonded_atom in attacking_bonded_atoms:
                for second_bonded_atom in attacking_bonded_atoms:
                    if bonded_atom != second_bonded_atom:
                        for third_bonded_atom in attacking_bonded_atoms:
                            if third_bonded_atom not in (bonded_atom, second_bonded_atom):
                                bond_vector = attacking_coords - reac_complex_coords[bonded_atom]
                                second_bond_vector = attacking_coords - reac_complex_coords[second_bonded_atom]
                                third_bond_vector = attacking_coords - reac_complex_coords[third_bonded_atom]
                                first_cross_product = np.cross(bond_vector, second_bond_vector)
                                second_cross_product = np.cross(bond_vector, third_bond_vector)
                                normed_first_cross_product = first_cross_product / np.linalg.norm(first_cross_product)
                                normed_second_cross_product = second_cross_product / np.linalg.norm(second_cross_product)
                                theta = np.arccos(np.dot(normed_first_cross_product, normed_second_cross_product))
                                if tolerance < theta < (np.pi - tolerance):
                                    flat = False

            if flat:
                logger.info(f'Attacking atom ({attacking_atom}) has flat geometry')
                all_attack_vectors.append([normed_first_cross_product, True])
            else:
                logger.info('Attacking atom does not have flat geometry')
                avg_attack = np.average(attacking_bond_vectors, axis=0)
                normed_avg_attack = avg_attack / np.linalg.norm(avg_attack)
                all_attack_vectors.append([normed_avg_attack, False])
    logger.info('Getting average attack vector')
    # since the flat atom can attack from either side, check which is best with other attacking atom
    if len(all_attack_vectors) == 1:
        best_avg_attack_vector = all_attack_vectors[0][0]
    else:
        if all_attack_vectors[0][1]:
            theta1 = np.arccos(np.dot(all_attack_vectors[0][0], all_attack_vectors[1][0]))
            opposite_attack = np.negative(all_attack_vectors[0][0])
            theta2 = np.arccos(np.dot(opposite_attack, all_attack_vectors[1][0]))
            if theta2 < theta1:
                all_attack_vectors[0][0] = opposite_attack
        if all_attack_vectors[1][1]:
            theta1 = np.arccos(np.dot(all_attack_vectors[0][0], all_attack_vectors[1][0]))
            opposite_attack = np.negative(all_attack_vectors[1][0])
            theta2 = np.arccos(np.dot(all_attack_vectors[0][0], opposite_attack))
            if theta2 < theta1:
                all_attack_vectors[1][0] = opposite_attack
        best_attack_vectors = [all_attack_vectors[0]
                               [0], all_attack_vectors[1][0]]
        best_avg_attack_vector = np.average(best_attack_vectors, axis=0)
    if np.linalg.norm(best_avg_attack_vector) == 0:
        best_avg_attack_vector = all_attack_vectors[0][0]
    return best_avg_attack_vector / np.linalg.norm(best_avg_attack_vector)


def dot_func(angles, mol1_vecs, mol2_vecs):
    theta, phi, z = angles
    theta_rot_matrix = calc_rotation_matrix([1, 0, 0], theta)
    phi_rot_matrix = calc_rotation_matrix([0, 1, 0], phi)
    z_rot_matrix = calc_rotation_matrix([0, 0, 1], z)
    rot_matrix = np.matmul(theta_rot_matrix, np.matmul(phi_rot_matrix, z_rot_matrix))
    dots = 0
    for i, mol1_vec in enumerate(mol1_vecs):
        mol2_vec = mol2_vecs[i]
        rot_mol2_vec = np.matmul(rot_matrix, mol2_vec)
        dots += np.dot(mol1_vec, rot_mol2_vec)
    return dots


def rot_func(angles, vec, mol1_coords, mol2_coords):
    theta = angles[0]
    rot_matrix = calc_rotation_matrix(vec, theta)
    new_mol2_coords = mol2_coords.copy()
    for i in range(len(mol2_coords)):
        new_mol2_coords[i] = np.matmul(rot_matrix, mol2_coords[i])
    dist_mat = distance_matrix(mol1_coords, new_mol2_coords)
    energy = np.sum(np.power(dist_mat, -4))
    return energy


def get_attacked_atoms(bond_rearrangement):
    """For a bond rearrangement find the atom which is common to both a forming and breaking bond in a substitution
    reaction thus has been attacked. Exit if there are more than two

    Arguments:
        bond_rearrangement (object): bond rearrangement object

    Returns:
        list: list of attacked atoms
    """

    possible_attacked_atoms = []
    for fbond in bond_rearrangement.fbonds:
        for bbond in bond_rearrangement.bbonds:
            [possible_attacked_atoms.append(atom) for atom in fbond if atom in bbond if not atom in possible_attacked_atoms]

    return possible_attacked_atoms


def get_leaving_or_attacking_atoms(bbonds_or_fbonds, attacked_atom):
    """Get the atom that attaches the attacked atom to the rest of the molecule. Each attacked
    atom is only attacked by one atom, so returning immediately after finding one is fine

    Arguments:
        bbonds_or_fbonds (list(tuple)): List of either breaking or forming bonds
        attacked_atom (int): index of the attacked atom in the structure

    Returns:
        list: indices of the attacking atoms
    """

    other_atoms = []

    for bond in bbonds_or_fbonds:
        if bond[0] == attacked_atom:
            other_atoms.append(bond[1])
        if bond[1] == attacked_atom:
            other_atoms.append(bond[0])
    return other_atoms
