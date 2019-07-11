from .log import logger
from .config import Config
from .single_point import get_single_point_energy
from .bond_lengths import get_xyz_bond_list
from .geom import calc_distance_matrix
from autode.templates import ActiveAtomEnvironment
from autode.templates import TStemplate


def get_bonded_atom_labels_atom_i(xyzs, bond_list, atom_i):

    bonded_atom_labels = []

    for bond in bond_list:
        if atom_i in bond:
            if atom_i == bond[0]:
                bonded_atom_labels.append(xyzs[bond[1]][0])                     # Pick the non active bonded atom label
            if atom_i == bond[1]:
                bonded_atom_labels.append(xyzs[bond[0]][0])

    return bonded_atom_labels


def get_bond_list_non_active_atoms(active_bonds, xyzs):
    logger.info('Getting list of bonds not containing the active atoms')
    bond_list = get_xyz_bond_list(xyzs)
    active_atoms = [atom_id for bond in active_bonds for atom_id in bond]

    for bond in bond_list:
        if any([atom_id in active_atoms for atom_id in bond_list]):
            bond_list.remove(bond)

    return bond_list


def get_aaenv_dists(active_bonds, xyzs, distance_matrix):
    logger.info('Getting aaenv_dists')
    bond_list_non_active_atoms = get_bond_list_non_active_atoms(active_bonds, xyzs)

    aaenv_dists = {}

    for bond_ids in active_bonds:
        bond_aaenvs = []
        for atom_id in bond_ids:
            bonded_atom_labels = get_bonded_atom_labels_atom_i(xyzs, bond_list_non_active_atoms, atom_id)

            bond_aaenvs.append(ActiveAtomEnvironment(atom_label=xyzs[atom_id][0],
                                                     bonded_atom_labels=bonded_atom_labels))

        aaenv_dists[tuple(bond_aaenvs)] = distance_matrix[bond_ids[0], bond_ids[1]]

    return aaenv_dists


class TS(object):

    def save_ts_template(self):
        logger.info('Saving TS template')
        aaenv_dists = get_aaenv_dists(self.active_bonds, self.xyzs, self.distance_matrix)

        ts_template = TStemplate(aaenv_dists, reaction_class=self.reaction_class, solvent=self.solvent,
                                 charge=self.charge, mult=self.mult)
        ts_template.save_object()
        logger.info('Saved TS template')

    def single_point(self):
        self.energy = get_single_point_energy(self, keywords=Config.sp_keywords, n_cores=Config.n_cores)

    def is_true_ts(self):
        if len(self.imag_freqs) == 1:
            return True
        else:
            return False

    def __init__(self, imag_freqs=None, xyzs=None, energy=None, name='TS', solvent=None, charge=0, mult=1,
                 converged=True, active_bonds=None, reaction_class=None):
        """
        Generate a TS object
        :param imag_freqs: (list) List of imaginary frequencies given as negative value
        """
        logger.info('Generating a TS object for {}'.format(name))

        self.name = name
        self.xyzs = xyzs
        self.energy = energy
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
        self.converged = converged
        self.imag_freqs = imag_freqs

        self.active_bonds = active_bonds
        self.reaction_class = reaction_class

        if xyzs is not None and active_bonds is not None and reaction_class is not None:
            self.distance_matrix = calc_distance_matrix(self.xyzs)
            self.save_ts_template()
