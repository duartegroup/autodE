from autode.log import logger
from autode.config import Config
from autode.geom import calc_distance_matrix
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix
from autode.atoms import get_vdw_radii
from autode import mol_graphs
from autode.transition_states.templates import TStemplate
from autode.calculation import Calculation
from autode.transition_states.ts_conformers import rot_bond
from autode.transition_states.rot_fragments import RotFragment
from autode.transition_states.ts_guess import TSguess
from autode.transition_states.optts import get_ts
from autode.transition_states.ts_conformers import TSConformer
import numpy as np


class TS(TSguess):

    def make_graph(self):
        logger.info('Making TS graph with \'active\' edges')

        full_graph = mol_graphs.make_graph(self.xyzs, n_atoms=len(self.xyzs))
        distance_matrix = calc_distance_matrix(self.xyzs)

        for bond in self.active_bonds:
            atom_i, atom_j = bond
            full_graph.add_edge(atom_i, atom_j, active=True,
                                weight=distance_matrix[atom_i, atom_j])

        nodes_to_keep = self.active_atoms.copy()
        for edge in full_graph.edges():
            node_i, node_j = edge
            if node_i in self.active_atoms:
                nodes_to_keep.append(node_j)
            if node_j in self.active_atoms:
                nodes_to_keep.append(node_i)

        self.graph = full_graph

        graph_with_fbonds = self.graph.copy()
        for bond in self.active_bonds:
            graph_with_fbonds.add_edge(*bond)

        self.graph_with_fbonds = graph_with_fbonds

        nodes_list = list(full_graph.nodes()).copy()
        truncated_graph = full_graph.copy()
        [truncated_graph.remove_node(
            node) for node in nodes_list if node not in nodes_to_keep]
        self.truncated_graph = truncated_graph

    def save_ts_template(self, folder_path=None):
        """
        Save a transition state template containing the active bond lengths, solvent and charge in folder_path

        :param folder_path: (str) folder to save the TS template to
        :return:
        """
        logger.info('Saving TS template')
        try:
            ts_template = TStemplate(self.truncated_graph, reaction_class=self.reaction_class, solvent=self.solvent,
                                     charge=self.charge, mult=self.mult)
            ts_template.save_object(folder_path=folder_path)
            logger.info('Saved TS template')

        except (ValueError, AttributeError):
            logger.error('Could not save TS template')

    def get_atom_label(self, atom_i):
        return self.xyzs[atom_i][0]

    def get_dist_consts(self):
        dist_const_dict = {}
        for bond in self.active_bonds:
            atom1, atom2 = bond
            coords = self.get_coords()
            bond_length = np.linalg.norm(coords[atom1] - coords[atom2])
            dist_const_dict[bond] = bond_length
        self.dist_consts = dist_const_dict

    def is_true_ts(self):
        if len(self.imag_freqs) == 1:
            return True
        else:
            return False

    def single_point(self, method=None):
        logger.info(f'Running single point energy evaluation of {self.name}')

        sp = Calculation(name=self.name + '_sp', molecule=self, method=self.method if method is None else method,
                         keywords=self.method.sp_keywords, n_cores=Config.n_cores, max_core_mb=Config.max_core)
        sp.run()
        self.energy = sp.get_energy()

    def get_rotatable_bonds(self):
        """Looks to see if each bond is worth rotating, i.e not methyl, trifluoromethyl or linear

        Returns:
            list -- list of bond tuples
        """
        logger.info('Finding the bonds worth rotating')
        bonds = list(self.graph_with_fbonds.edges)
        bonds_worth_rotating = []
        coords = self.get_coords()

        cycles = mol_graphs.find_cycle(self.graph_with_fbonds)

        ring_atoms = {}

        for potential_bond in bonds:
            suitable_bond = True
            half_suitable_bond = True

            # don't rotate pi bonds
            if self.pi_bonds is not None:
                if potential_bond in self.pi_bonds:
                    suitable_bond = False

            # don't rotate rings
            for cycle in cycles:
                if potential_bond[0] in cycle and potential_bond[1] in cycle:
                    suitable_bond = False
                    for atom in potential_bond:
                        if not atom in ring_atoms.keys():
                            bonded_atoms = self.get_bonded_atoms_to_i(atom)
                            non_ring_atoms = [
                                bonded_atom for bonded_atom in bonded_atoms if not bonded_atom in cycle]
                            two_rings = False
                            for non_ring_atom in non_ring_atoms:
                                for second_cycle in cycles:
                                    if non_ring_atom in second_cycle:
                                        two_rings = True
                                        break
                            if two_rings:
                                break
                            if len(non_ring_atoms) > 1:
                                if not (all(len(self.get_bonded_atoms_to_i(atom_i)) == 1 for atom_i in non_ring_atoms) and all(self.get_atom_label(atom_i) == self.get_atom_label(non_ring_atoms[0]) for atom_i in non_ring_atoms)):
                                    ring_bonded_atoms = [
                                        bonded_atom for bonded_atom in bonded_atoms if bonded_atom in cycle]
                                    ring_bonds = [(atom, ring_atom)
                                                  for ring_atom in ring_bonded_atoms]
                                    ring_atoms[atom] = ring_bonds
                    break

            if not suitable_bond:
                continue

            for index, atom in enumerate(potential_bond):
                other_bond_atom = potential_bond[1-index]
                bonded_atoms = self.get_bonded_atoms_to_i(atom)
                bonded_atoms.remove(other_bond_atom)

                # don't rotate terminal atoms
                if len(bonded_atoms) == 0:
                    suitable_bond = False
                    break

                # don't rotate linear
                if len(bonded_atoms) == 1:
                    bond_vector1 = coords[atom] - coords[other_bond_atom]
                    normed_bond_vector1 = bond_vector1 / \
                        np.linalg.norm(bond_vector1)
                    bond_vector2 = coords[bonded_atoms[0]] - coords[atom]
                    normed_bond_vector2 = bond_vector2 / \
                        np.linalg.norm(bond_vector2)
                    theta = np.arccos(
                        np.dot(normed_bond_vector1, normed_bond_vector2))
                    if theta < 0.09:
                        # have 3 colinear atoms, no need to rotate both bonds
                        if ((atom, bonded_atoms[0]) in bonds_worth_rotating) or ((bonded_atoms[0], atom) in bonds_worth_rotating):
                            suitable_bond = False
                            break
                        # want both atoms to have linear bonds, otherwise it is worth rotating
                        if not half_suitable_bond:
                            suitable_bond = False
                            break
                        half_suitable_bond = False

                # don't rotate methyl like
                if all(len(self.get_bonded_atoms_to_i(atom_i)) == 1 for atom_i in bonded_atoms) and all(self.get_atom_label(atom_i) == self.get_atom_label(bonded_atoms[0]) for atom_i in bonded_atoms):
                    suitable_bond = False
                    break

            if suitable_bond:
                bonds_worth_rotating.append(potential_bond)

        logger.info(
            f'Found {len(bonds_worth_rotating)} bond(s) worth rotating')
        logger.info(f'Found {len(ring_atoms)} ring atom(s) worth rotating')

        if len(ring_atoms) > 0:
            self.ring_atoms = ring_atoms

        self.rotatable_bonds = bonds_worth_rotating

    def get_central_bond(self):
        logger.info('Getting central bond')
        split_graphs_indices = [mol_graphs.split_mol_across_bond(
            self.graph_with_fbonds, [bond]) for bond in self.rotatable_bonds]
        smallest_difference = len(self.xyzs)
        for index, split in enumerate(split_graphs_indices):
            difference = abs(len(split[0]) - len(split[1]))
            if difference < smallest_difference:
                smallest_difference = difference
                central_bond = self.rotatable_bonds[index]
        logger.info(f'Found central bond {central_bond}')
        self.central_bond = central_bond

    def decompose(self):
        logger.info('Decomposing molecule into rotatable fragments')

        rot_bonds = self.rotatable_bonds.copy()
        rot_bonds.remove(self.central_bond)

        base_atom = self.central_bond[0]

        frag_level = 0
        rot_frags = []
        parent_graph = self.graph.copy()
        atoms_in_frags = []
        while rot_bonds:

            split_graphs_indices = [mol_graphs.split_mol_across_bond(
                parent_graph, [bond]) for bond in rot_bonds]
            largest_difference = None
            edge_bonds = []
            all_fragment_atoms = []
            for index, split in enumerate(split_graphs_indices):
                if base_atom in split[0]:
                    difference = len(split[0]) - len(split[1])
                    fragment_atoms = split[1]
                else:
                    difference = len(split[1]) - len(split[0])
                    fragment_atoms = split[0]
                if largest_difference is None:
                    largest_difference = difference
                    edge_bonds = [rot_bonds[index]]
                    all_fragment_atoms = [fragment_atoms]
                elif difference > largest_difference:
                    largest_difference = difference
                    edge_bonds = [rot_bonds[index]]
                    all_fragment_atoms = [fragment_atoms]
                elif difference == largest_difference:
                    edge_bonds.append(rot_bonds[index])
                    all_fragment_atoms.append(fragment_atoms)
            for edge_bond in edge_bonds:
                rot_bonds.remove(edge_bond)
            for edge_bond, frag_atoms in zip(edge_bonds, all_fragment_atoms):
                rot_frags.append(RotFragment(ts=self, rot_bond=edge_bond, dist_consts=self.dist_consts, atoms=frag_atoms,
                                             base_atom=base_atom, parent_graph=parent_graph, level=frag_level, all_rot_bonds=rot_bonds))
                atoms_in_frags += frag_atoms
            all_bonds_split_graphs = mol_graphs.split_mol_across_bond(
                parent_graph, edge_bonds, return_graphs=True)
            for graph in all_bonds_split_graphs:
                if base_atom in list(graph.nodes):
                    parent_graph = graph
                    break
            frag_level += 1
        rot_frags.append(RotFragment(ts=self, rot_bond=self.central_bond, level=frag_level, dist_consts=self.dist_consts, atoms=[
                         atom for atom in range(len(self.xyzs)) if atom not in atoms_in_frags]))
        frag_level += 1

        logger.info(f'Made {len(rot_frags)} rotatable fragments')

        # order frags by frag_level
        ordered_frags = []
        for i in range(frag_level):
            for frag in rot_frags:
                if frag.level == i:
                    ordered_frags.append(frag)

        self.rot_frags = ordered_frags

    def rotate(self, hlevel):
        # go from the outside in
        if self.ring_atoms is not None:
            logger.info('Rotating ring atoms')
            self.rotate_ring_atoms(hlevel)

        logger.info('Rotating fragments')
        for frag in self.rot_frags:
            frag.rotate(hlevel)

    def rotate_ring_atoms(self, hlevel, n_rotations=6):
        theta = np.pi * 2 / n_rotations
        for ring_atom, ring_bonds in self.ring_atoms.items():
            fragments = mol_graphs.split_mol_across_bond(
                self.graph_with_fbonds, ring_bonds)
            for fragment in fragments:
                if ring_atom in fragment:
                    atoms_to_rotate = fragment
                    break
            coords = self.get_coords()
            coords = coords - coords[ring_atom]
            rot_axis = coords[ring_bonds[0][1]] - coords[ring_bonds[1][1]]
            rot_matrix = calc_rotation_matrix(rot_axis, theta)
            confs = []
            for i in range(n_rotations):
                close_atoms = []
                for atom in atoms_to_rotate:
                    coords[atom] = np.matmul(rot_matrix, coords[atom])
                # check if any atoms are close
                for atom in range(len(coords)):
                    for other_atom in range(len(coords)):
                        if atom != other_atom:
                            distance_threshold = get_vdw_radii(
                                self.xyzs[atom][0]) + get_vdw_radii(self.xyzs[other_atom][0]) - 1.2
                            distance = np.linalg.norm(
                                coords[atom] - coords[other_atom])
                            if distance < distance_threshold:
                                close_atoms.append((atom, other_atom))
                    xyzs = coords2xyzs(coords, self.xyzs)
                confs.append(TSConformer(name=self.name + f'_rot_ring_{ring_atom}_{i}', close_atoms=close_atoms, xyzs=xyzs, rot_frags=self.rot_frags,
                                         dist_consts=self.dist_consts, solvent=self.solvent, charge=self.charge, mult=self.mult))

            [conf.optimise(hlevel) for conf in confs]

            logger.info('Setting TS xyzs as lowest energy xyzs')

            lowest_energy = None

            for conf in confs:
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
                self.xyzs = lowest_energy_conf.xyzs
                self.energy = lowest_energy_conf.energy

    def opt_ts(self, hlevel):
        name = self.name
        if hlevel:
            self.name += '_hlevel'
        else:
            self.name += '_llevel'
        opt = Calculation(name=self.name + '_opt', molecule=self, method=self.method, keywords=self.method.opt_keywords,
                          n_cores=Config.n_cores, max_core_mb=Config.max_core, distance_constraints=self.dist_consts)
        opt.run()
        self.energy = opt.get_energy()
        self.xyzs = opt.get_final_xyzs()

        ts_conf_get_ts_output = get_ts(self)
        if ts_conf_get_ts_output is None:
            return None
        self.converged = ts_conf_get_ts_output[1]
        self.name = name
        return self

    def do_conformers(self, hlevel=False):
        self.get_rotatable_bonds()
        if len(self.rotatable_bonds) == 0 and len(self.ring_atoms) == 0:
            logger.info('No bonds to rotate')
            return self
        self.get_central_bond()
        self.decompose()
        self.rotate(hlevel)
        return self.opt_ts(hlevel)

    def __init__(self, ts_guess, name='TS', converged=True):
        logger.info(f'Generating a TS object for {name}')

        self.name = name
        self.solvent = ts_guess.solvent
        self.charge = ts_guess.charge
        self.mult = ts_guess.mult
        self.converged = converged
        self.method = ts_guess.method
        self.pi_bonds = ts_guess.pi_bonds

        self.reactant = ts_guess.reactant
        self.product = ts_guess.product

        self.imag_freqs, self.xyzs, self.energy = ts_guess.get_imag_frequencies_xyzs_energy()
        self.n_atoms = len(self.xyzs)

        self.active_bonds = ts_guess.active_bonds
        self.active_atoms = list(
            set([atom_id for bond in self.active_bonds for atom_id in bond]))
        self.reaction_class = ts_guess.reaction_class

        self.graph = None
        self.truncated_graph = None
        self.graph_with_fbonds = None
        self.make_graph()

        self.rotatable_bonds = []
        self.ring_atoms = None
        self.central_bond = None
        self.rot_fragments = []

        self.dist_consts = None
        self.get_dist_consts()

        self.conformers = None
        self.n_conformers = None

        if Config.make_ts_template:
            self.save_ts_template()

        self.optts_converged = False
        self.optts_nearly_converged = False
        self.optts_calc = None
        self.hess_calc = None

        self.calc_failed = False
