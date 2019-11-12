from autode.mol_graphs import get_pathway
from autode.log import logger
from autode.transition_states.ts_conformers import rot_bond
from autode.transition_states.ts_conformers import TSConformer
import numpy as np


class RotFragment:

    def get_rot_pathway(self):
        """Gets the longest possible list of rotatable bonds from the beginning of 
        the fragment to the base of the parent fragment
        """
        if self.rot_bond[0] in self.atoms:
            start_atom = self.rot_bond[0]
        else:
            start_atom = self.rot_bond[1]
        logger.info(
            f'Getting rotational pathway from {start_atom} to {self.base_atom}')
        paths = get_pathway(self.parent_graph, start_atom, self.base_atom)
        max_rot_bonds_in_path = 0
        for path in paths:
            rot_path = []
            for bond in self.all_rot_bonds:
                if bond in path or (bond[1], bond[0]) in path:
                    rot_path.append(bond)
            if len(rot_path) > max_rot_bonds_in_path:
                max_rot_bonds_in_path = len(rot_path)
                rot_pathway = rot_path
        self.rot_pathway = rot_pathway

    def avoid_group(self, conf, other_frag, no_attempts=2):
        cart_consts = [i for i in conf.cart_consts if not i in self.atoms]
        for atom in self.ts.active_atoms:
            if not atom in cart_consts:
                cart_consts.append(atom)
        logger.info(f'Trying to fix clash by moving atom(s) {self.atoms}')
        if self.rot_pathway is None:
            logger.error('No rotational pathway, cannot fix clash')
            return None
        if len(self.rot_pathway) < no_attempts:
            no_attempts = len(self.rot_pathway)
        rot_confs = []
        for i in range(no_attempts):
            rotating_bond = self.rot_pathway[i]
            xyzs_backup = self.ts.xyzs
            self.ts.xyzs = conf.xyzs
            logger.info(
                f'Trying to fix clash by rotating bond {rotating_bond}')
            xyzs_close_atoms = rot_bond(self.ts, rotating_bond)
            self.ts.xyzs = xyzs_backup
            for i, (xyzs, close_atoms) in enumerate(xyzs_close_atoms):
                if not close_atoms:
                    rot_confs.append(TSConformer(name=conf.name + f'_rot_{self.rot_bond[0]}_{self.rot_bond[1]}_{i}', close_atoms=close_atoms, xyzs=xyzs,
                                                 rot_frags=self.ts.rot_frags, cart_consts=cart_consts, solvent=self.ts.solvent, charge=self.ts.charge, mult=self.ts.mult))
                else:
                    fixed_clash = True
                    for (atom1, atom2) in close_atoms:
                        if (atom1 in self.atoms and atom2 in other_frag.atoms) or (atom2 in self.atoms and atom1 in other_frag.atoms):
                            # check if the groups are now apart
                            fixed_clash = False
                        if not atom1 in conf.close_atoms or not atom2 in conf.close_atoms:
                            # check if it has made other atoms close
                            fixed_clash = False
                    if fixed_clash:
                        rot_confs.append(TSConformer(name=conf.name + f'_rot_{self.rot_bond[0]}_{self.rot_bond[1]}_{i}', close_atoms=close_atoms, xyzs=xyzs,
                                                     rot_frags=self.ts.rot_frags, cart_consts=cart_consts, solvent=self.ts.solvent, charge=self.ts.charge, mult=self.ts.mult))
            if len(rot_confs) > 1:
                logger.info('Multiple fixes found, getting lowest energy one')
                [conf.optimise() for conf in rot_confs]
                lowest_energy = None
                for conf in rot_confs:
                    if conf.energy is None or conf.xyzs is None:
                        continue
                    if lowest_energy is None:
                        lowest_energy = conf.energy
                        lowest_energy_conf = conf
                    elif conf.energy <= lowest_energy:
                        lowest_energy_conf = conf
                        lowest_energy = conf.energy
                    else:
                        pass
                return lowest_energy_conf.xyzs
            elif len(rot_confs) == 1:
                logger.info('Found a single rotation fixing the clash')
                return rot_confs[0].xyzs
            else:
                logger.info(
                    f'Rotating bond {rotating_bond} did not fix the clash')
        return None

    def rotate(self):
        cart_consts = [i for i in range(
            len(self.ts.xyzs)) if not i in self.atoms]
        for atom in self.ts.active_atoms:
            if not atom in cart_consts:
                cart_consts.append(atom)
        rot_confs_with_clashes = []
        rot_xyzs_close_atoms = rot_bond(mol=self.ts, bond=self.rot_bond)
        for i, (xyzs, close_atoms) in enumerate(rot_xyzs_close_atoms):
            rot_confs_with_clashes.append(TSConformer(name=self.ts.name + f'_rot_{self.rot_bond[0]}_{self.rot_bond[1]}_{i}', close_atoms=close_atoms,
                                                      xyzs=xyzs, rot_frags=self.ts.rot_frags, cart_consts=cart_consts, solvent=self.ts.solvent, charge=self.ts.charge, mult=self.ts.mult))

        rot_confs = []

        logger.info('Attempting to fix any close atoms')
        for conf in rot_confs_with_clashes:
            if conf.close_atoms is not None:
                conf.avoid_clash(self.rot_bond)
                if not conf.unfixed_clashes:
                    rot_confs.append(conf)
            else:
                rot_confs.append(conf)

        logger.info('Getting energy of each rotamer')

        [conf.optimise() for conf in rot_confs]

        lowest_energy = None

        logger.info('Setting TS xyzs as lowest energy xyzs')
        for conf in rot_confs:
            if conf.energy is None or conf.xyzs is None:
                continue
            if lowest_energy is None:
                lowest_energy = conf.energy
                lowest_energy_conf = conf
            elif conf.energy <= lowest_energy:
                lowest_energy_conf = conf
                lowest_energy = conf.energy
            else:
                pass

        if lowest_energy is not None:
            self.ts.xyzs = lowest_energy_conf.xyzs
            self.ts.energy = lowest_energy_conf.energy

    def __init__(self, ts, rot_bond, level, atoms=None, base_atom=None, parent_graph=None, all_rot_bonds=None):
        self.rot_bond = rot_bond
        self.ts = ts
        self.atoms = atoms
        self.base_atom = base_atom
        self.parent_graph = parent_graph
        self.level = level
        if all_rot_bonds is not None:
            self.all_rot_bonds = all_rot_bonds + [rot_bond]
        self.full_graph = ts.graph
        self.graph_with_fbonds = ts.graph_with_fbonds

        self.rot_pathway = None

        if self.base_atom is not None:
            self.get_rot_pathway()
