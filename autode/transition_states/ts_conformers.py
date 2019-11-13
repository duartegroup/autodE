from autode.log import logger
from autode.mol_graphs import split_mol_across_bond
from autode.geom import calc_rotation_matrix
from autode.geom import coords2xyzs
from autode.calculation import Calculation
from autode.config import Config
from autode.methods import get_hmethod
from autode.methods import get_lmethod
from autode.atoms import get_vdw_radii
import numpy as np


def rot_bond(mol, bond, number_rotations=12):
    theta = np.pi * 2 / number_rotations
    logger.info(f'Rotating bond {bond} in increments of {theta:.3f} radians')
    bond_atom_1, bond_atom_2 = bond
    coords = mol.get_coords().copy()
    split_indices = split_mol_across_bond(mol.graph_with_fbonds, [bond])
    coords -= coords[bond_atom_1]
    bond_vector = coords[bond_atom_1] - coords[bond_atom_2]
    rot_matrix = calc_rotation_matrix(bond_vector, theta)
    all_bonds = list(mol.graph.edges)

    if bond_atom_1 in split_indices[0]:
        atoms_to_shift = split_indices[0]
    else:
        atoms_to_shift = split_indices[1]

    rot_xyzs = []

    for i in range(number_rotations):
        close_atoms = []
        for atom in atoms_to_shift:
            coords[atom] = np.matmul(rot_matrix, coords[atom])
        # check if any atoms are close
        for atom in range(len(coords)):
            for other_atom in range(len(coords)):
                if (atom, other_atom) in all_bonds or (other_atom, atom) in all_bonds:
                    continue
                if atom != other_atom:
                    distance_threshold = get_vdw_radii(
                        mol.xyzs[atom][0]) + get_vdw_radii(mol.xyzs[other_atom][0]) - 1.2
                    distance = np.linalg.norm(
                        coords[atom] - coords[other_atom])
                    if distance < distance_threshold:
                        close_atoms.append((atom, other_atom))
        xyzs = coords2xyzs(coords, mol.xyzs)
        rot_xyzs.append((xyzs, close_atoms))
    return rot_xyzs


class TSConformer():

    def optimise(self, method=None):
        logger.info(f'Running optimisation of {self.name}')

        dist_consts = None
        cart_consts = None

        if method is None:
            method = get_lmethod()
            dist_consts = self.dist_consts
        else:
            cart_consts = self.cart_consts

        opt = Calculation(name=self.name + '_opt', molecule=self, method=method, keywords=method.opt_keywords,
                          n_cores=Config.n_cores, max_core_mb=Config.max_core, opt=True, distance_constraints=dist_consts, cartesian_constraints=cart_consts, constraints_already_met=True)
        opt.run()

        if opt.terminated_normally:
            self.energy = opt.get_energy()
            self.xyzs = opt.get_final_xyzs()
            with open('all_confs.xyz', 'a') as xyz_file:
                print(len(self.xyzs), '\n 0', file=xyz_file)
                [print('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(
                    *line), file=xyz_file) for line in self.xyzs]
        else:
            self.xyzs = None
            self.energy = None

    def avoid_clash(self, current_rot_bond):
        unfixed_clashes = False
        for (atom1, atom2) in self.close_atoms:
            logger.info(
                f'Trying to fix clash between atoms {atom1} and {atom2}')
            # check still close
            distance_threshold = get_vdw_radii(
                self.xyzs[atom1][0]) + get_vdw_radii(self.xyzs[atom2][0]) - 1.2
            if np.linalg.norm(np.asarray(self.xyzs[atom1][1:]) - np.asarray(self.xyzs[atom2][1:])) > distance_threshold:
                logger.info('Clash already sorted, skipping')
                continue
            else:
                for frag in self.rot_frags:
                    if atom1 in frag.atoms:
                        atom1_frag = frag
                    if atom2 in frag.atoms:
                        atom2_frag = frag
                if atom1_frag.level > atom2_frag.level:
                    fixed_xyzs = atom1_frag.avoid_group(self, atom2_frag)
                else:
                    fixed_xyzs = atom2_frag.avoid_group(self, atom1_frag)
            # check if fixed
            if fixed_xyzs is not None:
                logger.info('Clash fixed')
                self.xyzs = fixed_xyzs
            else:
                logger.info('Clash could not be fixed, ignoring conformer')
                unfixed_clashes = True
                break
        self.unfixed_clashes = unfixed_clashes

    def __init__(self, name='conf', close_atoms=None, xyzs=None, rot_frags=None, dist_consts=None, cart_consts=None, energy=None, solvent=None, charge=0, mult=1):
        self.name = name
        self.close_atoms = close_atoms if close_atoms is not [] else None
        self.xyzs = xyzs
        self.rot_frags = rot_frags
        self.dist_consts = dist_consts
        self.cart_consts = cart_consts
        self.n_atoms = len(xyzs) if xyzs is not None else None
        self.energy = energy
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
        self.unfixed_clashes = False
