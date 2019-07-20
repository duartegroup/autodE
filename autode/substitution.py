import numpy as np
from .log import logger
from .bond_lengths import get_xyz_bond_list
from .geom import xyz2coord
from .geom import calc_rotation_matrix


def get_complex_xyzs_translated_rotated(reac_complex, reac1, bond_rearrangement, shift_factor=2.0):
    logger.info('Translating reactant atoms into reactive complex')
    fbond, bbond = bond_rearrangement.fbonds[0], bond_rearrangement.bbonds[0]
    logger.info('Have assumed a dominant breaking bond of {}'.format(bond_rearrangement.bbonds[0]))

    nuc_atom, att_atom = sorted(fbond)        # will be smallest number in forming bond in the complex as reac1 is first
    nuc_atom_coords, att_atom_coords = xyz2coord(reac_complex.xyzs[nuc_atom]),  xyz2coord(reac_complex.xyzs[att_atom])

    att_lg_vec = get_attacked_atom_leaving_group_vector(reac_complex, bbond, fbond)

    complex_coords = xyz2coord(reac_complex.xyzs)
    for i in range(reac1.n_atoms, reac_complex.n_atoms):
        complex_coords[i] -= att_atom_coords - nuc_atom_coords
    for i in range(reac1.n_atoms):
        complex_coords[i] += shift_factor * att_lg_vec

    attack_vector = get_attack_vector_nuc_atom(reac1.xyzs, nuc_atom)

    if reac1.n_atoms > 1:
        logger.info('Rotating into best 180 degree attack')
        cos_theta = np.dot(att_lg_vec, attack_vector) / (np.linalg.norm(att_lg_vec) * np.linalg.norm(attack_vector))
        normal_vec = np.cross(att_lg_vec, attack_vector) / np.linalg.norm(np.cross(att_lg_vec, attack_vector))
        rot_matrix = calc_rotation_matrix(axis=normal_vec, theta=np.arccos(cos_theta))

        for i in range(reac_complex.n_atoms):
            complex_coords -= complex_coords[nuc_atom]
            if i < reac1.n_atoms:
                complex_coords[i] = np.matmul(rot_matrix, complex_coords[i])

    return [[reac_complex.xyzs[i][0]] + complex_coords[i].tolist() for i in range(reac_complex.n_atoms)]


def get_attacked_atom_leaving_group_vector(reac_complex, bbond, fbond):
    logger.info('Getting attacked atom –– leaving group vector')
    if bbond[0] in fbond:
        return xyz2coord(reac_complex.xyzs[bbond[0]]) - xyz2coord(reac_complex.xyzs[bbond[1]])
    elif bbond[1] in fbond:
        return xyz2coord(reac_complex.xyzs[bbond[1]]) - xyz2coord(reac_complex.xyzs[bbond[0]])
    else:
        logger.critical('Forming and breaking bond doesn\'t involve one common atom')
        return exit()


def get_attack_vector_nuc_atom(reac_xyzs, nuc_atom_id):
    logger.info('Getting vector of attack')

    bond_list = get_xyz_bond_list(reac_xyzs)
    nuc_bonded_atoms = [bond[0] for bond in bond_list if bond[1] == nuc_atom_id]
    nuc_bonded_atoms += [bond[1] for bond in bond_list if bond[0] == nuc_atom_id]

    nuc_atom_coords = xyz2coord(reac_xyzs[nuc_atom_id])
    nuc_atom_bonded_vectors = [nuc_atom_coords - xyz2coord(reac_xyzs[i]) for i in nuc_bonded_atoms]

    attack_vector = np.zeros(3)
    for bonded_vector in nuc_atom_bonded_vectors:
        attack_vector += bonded_vector / float(len(nuc_bonded_atoms))

    return attack_vector
