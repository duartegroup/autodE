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
    prod_complex_coords = prod_complex.get_coords()

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
        bond_forming_atoms = (get_leaving_or_attacking_atoms(bond_rearrangement.fbonds, atom))
        attack_vector = get_normalised_bond_forming_vector(reac_complex, prod_complex, reac_complex_coords, prod_complex_coords, mapping, atom, bond_forming_atoms)
        if atom in atoms_to_shift:
            mol1_vectors.append(attack_vector)
            mol2_vectors.append(bond_breaking_vector)
            mol1_atoms.append(bond_forming_atoms)
            mol2_atoms.append([atom])
        else:
            mol1_vectors.append(bond_breaking_vector)
            mol2_vectors.append(attack_vector)
            mol1_atoms.append([atom])
            mol2_atoms.append(bond_forming_atoms)

    min_dot = 9999999.9
    best_a, best_b, best_c = None, None, None
    for _ in range(100):
        x0 = np.random.uniform(0.0, 2*np.pi, size=3)
        res = minimize(dot_product_func, x0=x0, args=(mol1_vectors, mol2_vectors), method='BFGS')
        if res.fun < min_dot:
            min_dot = res.fun
            best_a, best_b, best_c = res.x

    a_rot_matrix = calc_rotation_matrix([1, 0, 0], best_a)
    b_rot_matrix = calc_rotation_matrix([0, 1, 0], best_b)
    c_rot_matrix = calc_rotation_matrix([0, 0, 1], best_c)
    rot_matrix = np.matmul(a_rot_matrix, np.matmul(b_rot_matrix, c_rot_matrix))

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
        min_energy = 9999999.9
        best_theta = None
        mol1_coords = reac_complex_coords[[i for i in range(reac_complex.n_atoms) if not i in atoms_to_shift]]
        mol2_coords = reac_complex_coords[atoms_to_shift]
        for _ in range(100):
            x0 = np.random.uniform(0.0, 2*np.pi, size=1)
            res = minimize(rot_mol_func, x0=x0, args=(avg_mol1_vector, mol1_coords, mol2_coords), method='BFGS')
            if res.fun < min_energy:
                min_energy = res.fun
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


def get_normalised_bond_forming_vector(reac_complex, prod_complex, reac_complex_coords, prod_complex_coords, mapping, atom, bond_forming_atoms):
    """Get the vector the new bond should be formed along, by seeing the vector of this bond in the products

    Arguments:
        reac_complex (mol obj): reactant complex object
        prod_complex (mol obj): product complex object
        reac_complex_coords (np.array) -- coords of the reactant
        prod_complex_coords (np.array) -- coords of the product
        mapping (dict): mapping[reactant indices] = product indices
        atom (int) -- atom bonds are forming to
        bond_forming_atoms (list) -- atom bonds are forming from

    Returns:
        np.array -- vector to form bond along
    """
    all_bond_forming_vectors = []
    for bond_forming_atom in bond_forming_atoms:
        reactant_bonded_atoms = reac_complex.get_bonded_atoms_to_i(bond_forming_atom)
        reactant_relevant_atoms = [bond_forming_atom, atom]
        for bonded_atom in reactant_bonded_atoms:
            if not bonded_atom in reactant_relevant_atoms:
                reactant_relevant_atoms.append(bonded_atom)
        product_relevant_atoms = [mapping[reac_atom] for reac_atom in reactant_relevant_atoms]
        reac_relevant_coords = reac_complex_coords[reactant_relevant_atoms].copy()
        reac_relevant_coords -= reac_relevant_coords[0]
        prod_relevant_coords = prod_complex_coords[product_relevant_atoms].copy()
        prod_relevant_coords -= prod_relevant_coords[0]

        # orient product onto reactant
        min_dist = 9999999.9
        best_a, best_b, best_c = None, None, None
        for _ in range(100):
            x0 = np.random.uniform(0.0, 2*np.pi, size=3)
            res = minimize(min_dist_func, x0=x0, args=(reac_relevant_coords, prod_relevant_coords), method='BFGS')
            if res.fun < min_dist:
                min_dist = res.fun
                best_a, best_b, best_c = res.x

        a_rot_matrix = calc_rotation_matrix([1, 0, 0], best_a)
        b_rot_matrix = calc_rotation_matrix([0, 1, 0], best_b)
        c_rot_matrix = calc_rotation_matrix([0, 0, 1], best_c)
        rot_matrix = np.matmul(a_rot_matrix, np.matmul(b_rot_matrix, c_rot_matrix))

        for i in range(len(prod_relevant_coords)):
            prod_relevant_coords[i] = np.matmul(rot_matrix, prod_relevant_coords[i])

        # get bond vector from product in orientation of reactant
        bond_forming_vector = prod_relevant_coords[1] - prod_relevant_coords[0]
        all_bond_forming_vectors.append(bond_forming_vector/np.linalg.norm(bond_forming_vector))

    avg_bond_forming_vector = np.average(all_bond_forming_vectors, axis=0)

    return avg_bond_forming_vector / np.linalg.norm(avg_bond_forming_vector)


def dot_product_func(angles, mol1_vecs, mol2_vecs):
    """Rotates mol2_vecs by the angles, the returns the sum of the dot products of mol1_vecs[i] with the rotated mol2_vecs[i]

    Arguments:
        angles (tuple) -- a,b,c angle
        mol1_vecs (list) -- list of vectors
        mol2_vecs (list) -- list of vectors

    Returns:
        float -- sum of the dot products
    """
    a, b, c = angles
    a_rot_matrix = calc_rotation_matrix([1, 0, 0], a)
    b_rot_matrix = calc_rotation_matrix([0, 1, 0], b)
    c_rot_matrix = calc_rotation_matrix([0, 0, 1], c)
    rot_matrix = np.matmul(a_rot_matrix, np.matmul(b_rot_matrix, c_rot_matrix))
    dots = 0
    for i, mol1_vec in enumerate(mol1_vecs):
        mol2_vec = mol2_vecs[i]
        rot_mol2_vec = np.matmul(rot_matrix, mol2_vec)
        dots += np.dot(mol1_vec, rot_mol2_vec)
    return dots


def rot_mol_func(angles, vec, mol1_coords, mol2_coords):
    """Rotates mol2_coords, then returns the sum of the distances^-4 between all pairs of mol1 atoms and mol2 atoms

    Arguments:
        angles (tuple) -- (angle to be rotated)
        vec (np.array) -- vector to be roated about
        mol1_coords (np.array) -- coords of mol1 atoms
        mol2_coords (np.array) -- coords of mol2 atoms

    Returns:
        float -- sum of the distances^-4 between all pairs of mol1 atoms and mol2 atoms
    """
    theta = angles[0]
    rot_matrix = calc_rotation_matrix(vec, theta)
    new_mol2_coords = mol2_coords.copy()
    for i in range(len(mol2_coords)):
        new_mol2_coords[i] = np.matmul(rot_matrix, mol2_coords[i])
    dist_mat = distance_matrix(mol1_coords, new_mol2_coords)
    energy = np.sum(np.power(dist_mat, -4))
    return energy


def min_dist_func(angles, mol1_coords, mol2_coords):
    """Rotates mol2_coords, then returns the average distance between mol1_atom[i] and mol2_atom[i]

    Arguments:
        angles (tuple) -- a,b,c angle
        mol1_coords (np.array) -- initial mol1_coords
        mol2_coords (np.array) -- initial mol2_coords

    Returns:
        float -- total distance
    """
    a, b, c = angles
    a_rot_matrix = calc_rotation_matrix([1, 0, 0], a)
    b_rot_matrix = calc_rotation_matrix([0, 1, 0], b)
    c_rot_matrix = calc_rotation_matrix([0, 0, 1], c)
    rot_matrix = np.matmul(a_rot_matrix, np.matmul(b_rot_matrix, c_rot_matrix))
    total_dist = 0
    new_mol2_coords = mol2_coords.copy()
    for i in range(1, len(mol2_coords)):
        new_mol2_coords[i] = np.matmul(rot_matrix, mol2_coords[i])
        total_dist += np.linalg.norm(mol1_coords[i] - new_mol2_coords[i])
    return total_dist


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
