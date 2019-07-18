import numpy as np
from .log import logger
from .geom import calc_distance_matrix
from .molecule import Molecule
from .rearrangement import get_possible_forming_bonds
from .rearrangement import generate_rearranged_graph
from .bond_lengths import get_xyz_bond_list
from .mol_graphs import is_isomorphic
from .geom import xyz2coord
from .geom import calc_rotation_matrix
from .dissociation import find_ts_breaking_bond
from .pes_2d import get_orca_ts_guess_2d
from .bond_lengths import get_avg_bond_length
from .reactions import Substitution
from .optts import get_ts
from .template_ts_guess import get_template_ts_guess


def find_ts(reaction):
    """
    Find a transition state for a substitution reaction i.e. reactant1 + reactant2 -> product1 + product2
    :param reaction:
    :return:
    """
    logger.info('Finding TS for a substitution reaction')

    reac_complex, prod_complex = get_reac_and_prod_complexes(reaction)
    fbond, bbond = get_forming_and_breaking_bonds(reac_complex, prod_complex)
    reac_complex.xyzs = get_complex_xyzs_translated_rotated(reac_complex, reaction.reacs[0], fbond, bbond)

    #                   in order of increasing computational expense
    for ts_find_func in [find_ts_from_template_subst, find_ts_breaking_bond, find_ts_2d_scan]:
        ts = ts_find_func(reac_complex, [bbond], [fbond])

        if ts is not None:
            return ts

    return None


def get_reac_and_prod_complexes(reaction):
    reac1, reac2 = reaction.reacs
    if reac1.mult == 2 and reac2.mult == 2:
        logger.warning('Diradical substitution reactions are not yet supported. Multiple spin states possible!')

    reac_complex = Molecule(name='reac_complex', xyzs=generate_complex_xyzs(reac1, reac2), solvent=reac1.solvent,
                            charge=(reac1.charge + reac2.charge), mult=(reac1.mult + reac2.mult - 1))

    prod1, prod2 = reaction.prods
    prod_complex = Molecule(name='prod_complex', xyzs=generate_complex_xyzs(prod1, prod2), solvent=prod1.solvent,
                            charge=(prod1.charge + prod2.charge), mult=(prod1.mult + prod2.mult - 1))

    return reac_complex, prod_complex


def get_complex_xyzs_translated_rotated(reac_complex, reac1, fbond, bbond, shift_factor=2.0):
    logger.info('Translating reactant atoms into reactive complex')

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


def generate_complex_xyzs(mol1, mol2, mol2_shift_ang=100):
    return mol1.xyzs + [xyz[:3] + [xyz[3] + mol2_shift_ang] for xyz in mol2.xyzs]


def get_forming_and_breaking_bonds(reac_complex, prod_complex):
    logger.info('Getting forming and breaking bonds for substitution reaction')
    forming_bonds, breaking_bonds = [], []

    possible_forming_bonds = [tuple(sorted((i, j))) for i in range(reac_complex.n_atoms) for j in
                              range(reac_complex.n_atoms) if i > j]

    possible_forming_bonds = get_possible_forming_bonds_over_fragments(reac_complex, possible_forming_bonds)
    possible_breaking_bonds = get_xyz_bond_list(xyzs=reac_complex.xyzs)

    for fbond in possible_forming_bonds:
        for bbond in possible_breaking_bonds:
            rearranged_graph = generate_rearranged_graph(reac_complex.graph, fbond, bbond)

            if is_isomorphic(rearranged_graph, prod_complex.graph):
                forming_bonds.append(fbond)
                breaking_bonds.append(bbond)

    if len(forming_bonds) == 1 and len(breaking_bonds) == 1:
        return tuple(forming_bonds[0]), tuple(breaking_bonds[0])
    else:
        logger.critical('Substitution made/broke >1 bond. Not implemented yet')
        return exit()


def get_possible_forming_bonds_over_fragments(reac_complex, possible_forming_bonds, inter_frag_cuttoff=50):
    logger.info('Getting possible inter-fragment forming bonds in reactant complex')
    inter_fragement_forming_bonds = []

    for fbond in possible_forming_bonds:
        if reac_complex.distance_matrix[fbond[0], fbond[1]] > inter_frag_cuttoff:
            inter_fragement_forming_bonds.append(fbond)

    return inter_fragement_forming_bonds


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


def find_ts_from_template_subst(reac_complex, fbonds, bbonds):
    ts_guess = get_template_ts_guess(mol=reac_complex, active_bonds=fbonds + bbonds, reaction_class=Substitution)
    if ts_guess.xyzs is not None:
        return get_ts(ts_guess)

    return None


def find_ts_2d_scan(reac_complex, fbonds, bbonds):
    ts_guess = get_ts_guess_2d_scan_subst(reac_complex, fbond=fbonds[0], bbond=bbonds[0])
    if ts_guess.xyzs is not None:
        return get_ts(ts_guess)

    return None


def get_ts_guess_2d_scan_subst(reac_complex, fbond, bbond, max_bond_dist_add=1.5):

    distance_matrix = calc_distance_matrix(reac_complex.xyzs)
    fbond_curr_dist = distance_matrix[fbond[0], fbond[1]]
    fbond_final_dist = get_avg_bond_length(atom_i_label=reac_complex.xyzs[fbond[0]][0],
                                           atom_j_label=reac_complex.xyzs[fbond[1]][0])

    bbond_curr_dist = reac_complex.distance_matrix[bbond[0], bbond[1]]
    bbond_final_dist = bbond_curr_dist + max_bond_dist_add
    return get_orca_ts_guess_2d(mol=reac_complex, bond_ids=[fbond, bbond], curr_dist1=fbond_curr_dist,
                                final_dist1=fbond_final_dist, curr_dist2=bbond_curr_dist, final_dist2=bbond_final_dist,
                                reaction_class=Substitution)
